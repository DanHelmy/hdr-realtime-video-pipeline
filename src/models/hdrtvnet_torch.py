import copy
import contextlib
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import struct
import sys
import threading
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hip_sdk_detection import detect_hip_sdk_windows
from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from models.hdrtvnet_modules.HG_Composite_arch import HG_Composite
_ROCM_INCLUDE_DIRS = []
try:
    from windows_runtime import configure_rocm_sdk_environment, rocm_sdk_include_dirs

    configure_rocm_sdk_environment()
    _ROCM_INCLUDE_DIRS = rocm_sdk_include_dirs()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_HG_WEIGHTS = os.path.join(
    _ROOT, "src", "models", "weights", "original", "HG.pt"
)
_TENSORRT_SOURCE_SUBDIR = "distilled"
_ORIGINAL_TENSORRT_SOURCE_SUBDIR = "tensorrt"
_TENSORRT_CALIBRATION_SUBDIR = "tensorrt_calibration"
_TENSORRT_ENGINE_METADATA_SCHEMA = "hdrtvnet_tensorrt_engine_v1"
_TENSORRT_SOURCE_SIGNATURE_VERSION = "trt_source_v2"
_TENSORRT_SOURCE_CHECKPOINT_SCHEMA = "hdrtvnet_tensorrt_source_v1"
_TENSORRT_DEFAULT_INT8_FP_EXPORT_PREFIXES: tuple[str, ...] = ()
_PORTABLE_INT8_CHECKPOINT_FORMAT = "portable_fake_quant_v1"
_PORTABLE_INT8_STATE_FORMAT = "native_fp32"
_FILE_FINGERPRINT_CACHE: dict[tuple[str, int, int], dict[str, object]] = {}
_TENSORRT_CALIBRATION_IMAGE_SCORE_CACHE: dict[tuple[str, int, int], tuple[float, int]] = {}
_TENSORRT_SOURCE_SIGNATURE_CACHE: str | None = None

_HAS_COMPILE = hasattr(torch, "compile")          # PyTorch >= 2.0
_HAS_CUDA_GRAPHS = hasattr(torch.cuda, "CUDAGraph")  # PyTorch >= 1.10
_IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
_IS_NVIDIA = torch.cuda.is_available() and not _IS_ROCM
if _IS_ROCM:
    warnings.filterwarnings(
        "ignore",
        message=r"Please use the new API settings to control TF32 behavior.*",
        category=UserWarning,
    )

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _patch_triton_amd_include_dirs() -> None:
    """Work around ROCm wheels where Triton misses _rocm_sdk_core/include."""
    if not _HAS_TRITON or os.name != "nt" or not _IS_ROCM:
        return
    try:
        from triton.backends.amd import driver as amd_driver
    except Exception:
        return
    include_dirs = getattr(amd_driver, "include_dirs", None)
    if not isinstance(include_dirs, list):
        return
    normalized = {os.path.normcase(os.path.normpath(str(p))) for p in include_dirs}
    for include_dir in _ROCM_INCLUDE_DIRS:
        key = os.path.normcase(os.path.normpath(str(include_dir)))
        if key not in normalized:
            include_dirs.append(str(include_dir))
            normalized.add(key)


def _patch_inductor_rocm_cluster_dims() -> None:
    """Add missing Triton cluster metadata expected by Torch 2.9 Inductor."""
    if not _HAS_TRITON or os.name != "nt" or not _IS_ROCM:
        return
    try:
        from torch._inductor.runtime import triton_heuristics
    except Exception:
        return

    cls = getattr(triton_heuristics, "TritonCompileResult", None)
    if cls is None or getattr(cls, "_hdrtvnet_rocm_cluster_patch", False):
        return
    original_make_launcher = getattr(cls, "make_launcher", None)
    if original_make_launcher is None:
        return

    def _make_launcher_with_cluster_dims(self):
        kernel = getattr(self, "kernel", None)
        metadata = getattr(kernel, "metadata", None)
        if metadata is not None and not hasattr(metadata, "cluster_dims"):
            try:
                fields = tuple(getattr(metadata, "_fields", ()))
                if fields and hasattr(metadata, "_asdict"):
                    values = dict(metadata._asdict())
                    values["cluster_dims"] = (1, 1, 1)
                    meta_cls = cls._kernel_metadata_cls(tuple(values.keys()))
                    kernel.metadata = meta_cls(**values)
                else:
                    # Last-resort path for non-namedtuple metadata objects.
                    setattr(metadata, "cluster_dims", (1, 1, 1))
            except Exception:
                pass
        return original_make_launcher(self)

    cls.make_launcher = _make_launcher_with_cluster_dims
    cls._hdrtvnet_rocm_cluster_patch = True


_patch_triton_amd_include_dirs()
_patch_inductor_rocm_cluster_dims()

if os.name == "nt" and _IS_ROCM:
    _HAS_HIP_SDK, _HIP_SDK_ROOT = detect_hip_sdk_windows()
else:
    _HAS_HIP_SDK, _HIP_SDK_ROOT = False, None

# Enable TF32 on Ampere+ NVIDIA GPUs. Keep ROCm out of this path to avoid
# PyTorch 2.9 TF32 deprecation warnings for an optimization AMD cannot use.
if _IS_NVIDIA and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _torch_compile_dynamic_setting():
    value = os.environ.get("HDRTVNET_COMPILE_DYNAMIC", "0")
    text = str(value).strip().lower()
    if text in {"auto", "none", "default"}:
        return None
    return _env_bool("HDRTVNET_COMPILE_DYNAMIC", False)


def _torch_compile_warmup_runs() -> int:
    try:
        value = int(os.environ.get("HDRTVNET_COMPILE_WARMUP_RUNS", "2"))
    except Exception:
        value = 2
    return min(10, max(1, value))


def _normalize_predequantize_setting(value):
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "auto").strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return "auto"


_DEFAULT_TORCH_COMPILE_MODE = "max-autotune"
_AUTO_ASSUME_ALIGNED_SHAPES = {
    (1920, 1080),
    (1280, 720),
}


def _use_channels_last_by_default() -> bool:
    # ROCm max-autotune can benchmark a different kernel set for NHWC tensors,
    # and on RDNA cards that is not always faster. Keep ROCm contiguous unless
    # explicitly opted in for A/B testing.
    if not _env_bool("HDRTVNET_CHANNELS_LAST", False):
        return False
    return True


def _assume_aligned_shapes_for_resolution(width: int, height: int) -> bool:
    value = os.environ.get("HDRTVNET_ASSUME_ALIGNED_SHAPES", "auto")
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    if text in {"force", "always"}:
        return True
    return (int(width), int(height)) in _AUTO_ASSUME_ALIGNED_SHAPES


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
        self.register_buffer("x_scale", torch.tensor(1.0, dtype=torch.float32))
        if asymmetric:
            self.register_buffer("x_zero", torch.tensor(0.0, dtype=torch.float32))

    def _apply(self, fn):
        x_scale = self.x_scale.detach().clone()
        x_zero = self.x_zero.detach().clone() if hasattr(self, "x_zero") else None
        super()._apply(fn)
        # Keep activation quantization ranges in FP32 even when the module is
        # moved to FP16 compute.  Some calibrated scales are below FP16 normal
        # range; downcasting them can collapse the whole eager INT8 path.
        self.x_scale.data = x_scale.to(device=self.x_scale.device, dtype=torch.float32)
        if hasattr(self, "x_zero") and x_zero is not None:
            self.x_zero.data = x_zero.to(device=self.x_zero.device, dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_asymmetric:
            # Asymmetric: map [x_zero, x_zero + x_scale*255] ' [0, 255]
            x_f = x.float()
            x_q = ((x_f - self.x_zero) / self.x_scale).round().clamp(0, 255)
            x_deq = (x_q * self.x_scale + self.x_zero).to(self.compute_dtype)
        else:
            # Symmetric: map [-x_scale*127, x_scale*127] ' [-128, 127]
            x_int8 = (x.float() / self.x_scale).round().clamp(-128, 127).to(torch.int8)
            x_deq = x_int8.to(self.compute_dtype) * self.x_scale.to(self.compute_dtype)
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
        self.register_buffer("x_scale", torch.tensor(1.0, dtype=torch.float32))
        if asymmetric:
            self.register_buffer("x_zero", torch.tensor(0.0, dtype=torch.float32))

    def _apply(self, fn):
        x_scale = self.x_scale.detach().clone()
        x_zero = self.x_zero.detach().clone() if hasattr(self, "x_zero") else None
        super()._apply(fn)
        self.x_scale.data = x_scale.to(device=self.x_scale.device, dtype=torch.float32)
        if hasattr(self, "x_zero") and x_zero is not None:
            self.x_zero.data = x_zero.to(device=self.x_zero.device, dtype=torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_asymmetric:
            x_f = x.float()
            x_q = ((x_f - self.x_zero) / self.x_scale).round().clamp(0, 255)
            x_deq = (x_q * self.x_scale + self.x_zero).to(self.compute_dtype)
        else:
            x_int8 = (x.float() / self.x_scale).round().clamp(-128, 127).to(torch.int8)
            x_deq = x_int8.to(self.compute_dtype) * self.x_scale.to(self.compute_dtype)
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
    dtype_name = "FP16" if compute_dtype == torch.float16 else "FP32"
    print(f"  Pre-dequantized {converted} INT8 layers -> native {dtype_name} "
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
        self.register_buffer(
            "x_min",
            x_zero.detach().float().reshape(()),
        )
        # Convert the existing asymmetric min/scale form into ONNX affine
        # zero-point form. Clamping keeps the graph valid for uint8 Q/DQ.
        zp = torch.round(-x_zero.detach().float() / self.x_scale).clamp(0, 255)
        self.register_buffer("x_zero_point", zp.to(torch.int32).reshape(()))

    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activation:
            return x
        scale = self.x_scale.to(device=x.device)
        if self.activation_asymmetric:
            # TensorRT CUDA requires signed INT8 Q/DQ with zero_point=0.
            # Preserve the calibrated asymmetric range by expanding the
            # symmetric scale around zero. This is exact only when the range
            # is zero-centered, but avoids saturation for one-sided ranges.
            x_min = self.x_min.to(device=x.device)
            x_max = x_min + scale * 255.0
            sym_scale = torch.clamp(
                torch.maximum(torch.abs(x_min), torch.abs(x_max)) / 127.0,
                min=1e-8,
            )
            zp_zero = torch.tensor(0, dtype=torch.int32, device=x.device)
            return torch.fake_quantize_per_tensor_affine(x, sym_scale, zp_zero, -128, 127)
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
        w = self._quantize_weight()
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
        self.register_buffer(
            "x_min",
            x_zero.detach().float().reshape(()),
        )
        zp = torch.round(-x_zero.detach().float() / self.x_scale).clamp(0, 255)
        self.register_buffer("x_zero_point", zp.to(torch.int32).reshape(()))

    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activation:
            return x
        scale = self.x_scale.to(device=x.device)
        if self.activation_asymmetric:
            x_min = self.x_min.to(device=x.device)
            x_max = x_min + scale * 255.0
            sym_scale = torch.clamp(
                torch.maximum(torch.abs(x_min), torch.abs(x_max)) / 127.0,
                min=1e-8,
            )
            zp_zero = torch.tensor(0, dtype=torch.int32, device=x.device)
            return torch.fake_quantize_per_tensor_affine(x, sym_scale, zp_zero, -128, 127)
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
        w = self._quantize_weight()
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class TRTFloatConv2d(nn.Module):
    """Conv2d export module for W8A16 layers as a plain FP TensorRT op."""

    def __init__(self, w8_mod):
        super().__init__()
        self.stride = w8_mod.stride
        self.padding = w8_mod.padding
        self.dilation = w8_mod.dilation
        self.groups = w8_mod.groups

        scale = getattr(w8_mod, "scale", None)
        if scale is None:
            scale = w8_mod.w_scale
        export_dtype = getattr(w8_mod, "compute_dtype", torch.float16)
        scale = scale.detach().float()
        weight = (
            w8_mod.weight_int8.detach().float() * scale.view(-1, 1, 1, 1)
        ).to(dtype=export_dtype)
        self.weight = nn.Parameter(weight, requires_grad=False)
        if w8_mod.bias is not None:
            self.bias = nn.Parameter(
                w8_mod.bias.detach().to(dtype=export_dtype).clone(),
                requires_grad=False,
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.conv2d(x, w, bias, self.stride, self.padding,
                        self.dilation, self.groups)


class TRTFloatLinear(nn.Module):
    """Linear export module for W8A16 layers as a plain FP TensorRT op."""

    def __init__(self, w8_mod):
        super().__init__()
        scale = getattr(w8_mod, "scale", None)
        if scale is None:
            scale = w8_mod.w_scale
        export_dtype = getattr(w8_mod, "compute_dtype", torch.float16)
        scale = scale.detach().float()
        weight = (
            w8_mod.weight_int8.detach().float() * scale.view(-1, 1)
        ).to(dtype=export_dtype)
        self.weight = nn.Parameter(weight, requires_grad=False)
        if w8_mod.bias is not None:
            self.bias = nn.Parameter(
                w8_mod.bias.detach().to(dtype=export_dtype).clone(),
                requires_grad=False,
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class _GlobalAvgPool2d(nn.Module):
    """TRT-compatible replacement for AdaptiveAvgPool2d(1) — exports as ReduceMean."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


def _patch_adaptive_avgpool_for_trt(model: nn.Module) -> None:
    """Replace AdaptiveAvgPool2d(output_size=1) in-place with ReduceMean-based equivalent.

    TensorRT 10.x does not support the SequenceEmpty ONNX op that the dynamo exporter
    emits for AdaptiveAvgPool2d when it is lowered through aten.as_strided.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool2d) and child.output_size in (1, (1, 1)):
            setattr(model, name, _GlobalAvgPool2d())
        else:
            _patch_adaptive_avgpool_for_trt(child)


def _tensorrt_fp_export_prefixes(
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    raw = os.environ.get("HDRTVNET_TRT_FP_EXPORT_PREFIXES")
    if raw is None:
        return tuple(default)
    if raw.strip().lower() in {"", "0", "false", "no", "none", "off"}:
        return ()
    prefixes = []
    for item in raw.split(","):
        prefix = item.strip().strip(".")
        if prefix:
            prefixes.append(prefix)
    return tuple(dict.fromkeys(prefixes))


def _tensorrt_int8_fp_export_prefixes() -> tuple[str, ...]:
    return _tensorrt_fp_export_prefixes(
        _TENSORRT_DEFAULT_INT8_FP_EXPORT_PREFIXES
    )


def _tensorrt_module_matches_prefix(name: str,
                                    prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.")
               for prefix in prefixes)


def _tensorrt_fp_export_policy_suffix() -> str:
    prefixes = _tensorrt_int8_fp_export_prefixes()
    if not prefixes:
        return ""
    labels = []
    for prefix in prefixes:
        label = re.sub(r"[^a-z0-9]+", "", prefix.lower())
        labels.append(label or "module")
    return f"_fp{'-'.join(labels)}v1"


def _convert_model_to_tensorrt_qdq(model: nn.Module) -> nn.Module:
    """Replace W8/W8A8 runtime wrappers with TensorRT-friendly Q/DQ modules."""
    converted_w8a8 = 0
    converted_w8a16 = 0
    converted_w8_fp = 0
    forced_w8a8_fp = 0
    fp_prefixes = _tensorrt_int8_fp_export_prefixes()
    for name, module in list(model.named_modules()):
        if not isinstance(module, (W8Conv2d, W8A8Conv2d, W8Linear, W8A8Linear)):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        force_fp_export = _tensorrt_module_matches_prefix(name, fp_prefixes)
        quantize_activation = (
            isinstance(module, (W8A8Conv2d, W8A8Linear)) and
            not force_fp_export
        )
        if quantize_activation and isinstance(module, W8A8Conv2d):
            replacement = TRTQDQConv2d(module, quantize_activation)
            converted_w8a8 += 1
        elif quantize_activation and isinstance(module, W8A8Linear):
            replacement = TRTQDQLinear(module, quantize_activation)
            converted_w8a8 += 1
        elif force_fp_export and isinstance(module, (W8Conv2d, W8A8Conv2d)):
            replacement = TRTFloatConv2d(module)
            converted_w8_fp += 1
            forced_w8a8_fp += int(isinstance(module, W8A8Conv2d))
        elif force_fp_export:
            replacement = TRTFloatLinear(module)
            converted_w8_fp += 1
            forced_w8a8_fp += int(isinstance(module, W8A8Linear))
        elif isinstance(module, (W8Conv2d, W8A8Conv2d)):
            replacement = TRTQDQConv2d(module, quantize_activation=False)
            converted_w8a16 += 1
        else:
            replacement = TRTQDQLinear(module, quantize_activation=False)
            converted_w8a16 += 1
        setattr(parent, parts[-1], replacement)
    if converted_w8a8 or converted_w8a16 or converted_w8_fp:
        print(
            "TensorRT Q/DQ export: "
            f"{converted_w8a8} W8A8 layers, "
            f"{converted_w8a16} W8A16 layers exported as weight Q/DQ, "
            f"{converted_w8_fp} layer(s) exported as FP"
        )
        if fp_prefixes:
            print(
                "TensorRT Q/DQ export policy: "
                f"FP prefixes={list(fp_prefixes)}, "
                f"forced W8A8 FP layers={forced_w8a8_fp}"
            )
    return model


def _convert_model_to_tensorrt_native_layers(model: nn.Module) -> nn.Module:
    """Replace W8/W8A8 wrappers with plain layers for TensorRT native INT8 PTQ."""
    converted_conv = 0
    converted_linear = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, (W8Conv2d, W8A8Conv2d, W8Linear, W8A8Linear)):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        if isinstance(module, (W8Conv2d, W8A8Conv2d)):
            replacement = TRTFloatConv2d(module)
            converted_conv += 1
        else:
            replacement = TRTFloatLinear(module)
            converted_linear += 1
        setattr(parent, parts[-1], replacement)
    if converted_conv or converted_linear:
        print(
            "TensorRT native INT8 export: "
            f"{converted_conv} Conv and {converted_linear} Linear W8 layer(s) "
            "exported as plain ONNX layers for TensorRT calibration"
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


def _is_portable_int8_checkpoint(checkpoint: dict) -> bool:
    """Return True for backend-neutral checkpoints with native FP state."""
    if not isinstance(checkpoint, dict):
        return False
    if checkpoint.get("checkpoint_format") == _PORTABLE_INT8_CHECKPOINT_FORMAT:
        return True
    return checkpoint.get("state_format") == _PORTABLE_INT8_STATE_FORMAT


def _scalar_float(value, default: float | None = None) -> float | None:
    """Read a scalar from Python, NumPy, or torch values without keeping tensors."""
    if value is None:
        return default
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return float(value.detach().float().reshape(-1)[0].item())
        return float(value)
    except Exception:
        return default


def _activation_qparams_from_checkpoint(checkpoint: dict) -> dict[str, dict[str, float]]:
    """Normalize portable activation qparams to {layer: {scale, zero}}."""
    normalized: dict[str, dict[str, float]] = {}
    raw_qparams = checkpoint.get("activation_qparams") or {}
    if isinstance(raw_qparams, dict):
        for name, entry in raw_qparams.items():
            if isinstance(entry, dict):
                scale = _scalar_float(entry.get("scale"))
                zero = _scalar_float(entry.get("zero"), 0.0)
            else:
                scale = _scalar_float(entry)
                zero = 0.0
            if scale is not None and math.isfinite(scale) and scale > 0.0:
                normalized[str(name)] = {
                    "scale": max(float(scale), 1e-8),
                    "zero": 0.0 if zero is None else float(zero),
                }

    raw_scales = checkpoint.get("activation_scales") or {}
    raw_zeros = checkpoint.get("activation_zero_points") or {}
    if isinstance(raw_scales, dict):
        for name, value in raw_scales.items():
            if str(name) in normalized:
                continue
            scale = _scalar_float(value)
            if scale is None or not math.isfinite(scale) or scale <= 0.0:
                continue
            zero = 0.0
            if isinstance(raw_zeros, dict):
                zero = _scalar_float(raw_zeros.get(name), 0.0) or 0.0
            normalized[str(name)] = {
                "scale": max(float(scale), 1e-8),
                "zero": float(zero),
            }
    return normalized


def _apply_portable_activation_qparams(model: nn.Module, checkpoint: dict) -> None:
    """Apply calibrated activation qparams after recreating W8A8 modules."""
    qparams = _activation_qparams_from_checkpoint(checkpoint)
    if not qparams:
        return

    expected = 0
    applied = 0
    for name, module in model.named_modules():
        if not isinstance(module, (W8A8Conv2d, W8A8Linear)):
            continue
        expected += 1
        entry = qparams.get(name)
        if not entry:
            continue
        module.x_scale.fill_(float(entry["scale"]))
        if hasattr(module, "x_zero"):
            module.x_zero.fill_(float(entry.get("zero", 0.0)))
        applied += 1
    if expected:
        print(f"  Applied portable activation scales: {applied}/{expected}")


def _apply_portable_weight_qparams(model: nn.Module, checkpoint: dict) -> None:
    """Restore exact INT8 weight tensors/scales when a portable checkpoint has them."""
    qparams = checkpoint.get("weight_qparams") or {}
    if not isinstance(qparams, dict) or not qparams:
        return

    expected = 0
    applied = 0
    for name, module in model.named_modules():
        if not isinstance(module, (W8Conv2d, W8Linear, W8A8Conv2d, W8A8Linear)):
            continue
        expected += 1
        entry = qparams.get(name)
        if not isinstance(entry, dict):
            continue
        weight_int8 = entry.get("weight_int8")
        scale = entry.get("scale")
        if not torch.is_tensor(weight_int8) or not torch.is_tensor(scale):
            continue
        module.weight_int8.copy_(
            weight_int8.to(device=module.weight_int8.device, dtype=module.weight_int8.dtype)
        )
        target_scale = getattr(module, "w_scale", None)
        if target_scale is None:
            target_scale = getattr(module, "scale", None)
        if target_scale is None:
            continue
        target_scale.copy_(
            scale.to(device=target_scale.device, dtype=target_scale.dtype)
        )
        applied += 1
    if expected:
        print(f"  Applied portable weight qparams: {applied}/{expected}")


def _get_gpu_info() -> str:
    """Return GPU name string, or empty if unavailable."""
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return ""


def _is_int8_checkpoint_path(path: str) -> bool:
    try:
        name = os.path.basename(str(path)).lower()
        resolved = pathlib.Path(path).resolve()
    except Exception:
        return False
    if not (name.endswith(".pt") and "_int8_" in name):
        return False
    parts = {part.lower() for part in resolved.parts}
    if _TENSORRT_SOURCE_SUBDIR.lower() in parts:
        return False
    if "original" in parts and _ORIGINAL_TENSORRT_SOURCE_SUBDIR.lower() in parts:
        return False
    try:
        if not os.path.isfile(path):
            return True
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return True
    if not isinstance(checkpoint, dict):
        return False
    if checkpoint.get("checkpoint_format") == _PORTABLE_INT8_CHECKPOINT_FORMAT:
        return True
    return bool(checkpoint.get("state_dict")) and str(
        checkpoint.get("quantization", "")
    ) in {"w8a8_full", "w8a8_mixed"}


def _is_tensorrt_source_path(path: str) -> bool:
    try:
        parts = {part.lower() for part in pathlib.Path(path).resolve().parts}
    except Exception:
        return False
    return (
        _TENSORRT_SOURCE_SUBDIR.lower() in parts
        or (
            "original" in parts
            and _ORIGINAL_TENSORRT_SOURCE_SUBDIR.lower() in parts
        )
    )


def _tensorrt_source_candidate_path(model_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(model_path)))
    path_obj = pathlib.Path(path)
    directory, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    source_family = "hr"
    original_family = "original" in {part.lower() for part in path_obj.parts}
    if stem.startswith("HR_original_int8_"):
        tag = stem[len("HR_original_int8_"):]
        mapped = {
            "mixed": "int8_mixed_ptq",
            "full": "int8_full_ptq",
        }.get(tag, f"int8_{tag}")
        name = f"HR_original_{mapped}{ext}"
    elif stem.startswith("HR_HG_original_int8_"):
        source_family = "hg"
        tag = stem[len("HR_HG_original_int8_"):]
        mapped = {
            "mixed": "int8_mixed_ptq",
            "full": "int8_full_ptq",
        }.get(tag, f"int8_{tag}")
        name = f"HG_original_{mapped}{ext}"
    elif stem.startswith("HR_int8_"):
        tag = stem[len("HR_int8_"):]
        mapped = {
            "mixed": "int8_mixed_ptq",
            "full": "int8_full_ptq",
        }.get(tag, f"int8_{tag}")
        name = f"HR_qfriendly_spatialmixglobal_{mapped}{ext}"
    elif stem.startswith("HR_HG_int8_"):
        source_family = "hg"
        tag = stem[len("HR_HG_int8_"):]
        mapped = {
            "mixed": "int8_mixed_ptq",
            "full": "int8_full_ptq",
        }.get(tag, f"int8_{tag}")
        name = f"HG_qfriendly_directh16_{mapped}{ext}"

    if path_obj.parent.name.lower() in {"hr", "hg"} and path_obj.parent.parent.name.lower() in {"int8", "pytorch_int8"}:
        weights_root = path_obj.parent.parent.parent
        if original_family:
            return str(
                weights_root
                / _ORIGINAL_TENSORRT_SOURCE_SUBDIR
                / source_family
                / name
            )
        return str(weights_root / _TENSORRT_SOURCE_SUBDIR / source_family / name)
    if os.path.basename(directory).lower() == "int8":
        directory = os.path.dirname(directory)
    return os.path.join(directory, _TENSORRT_SOURCE_SUBDIR, source_family, name)


def _hash_file_raw(path: str) -> dict[str, object]:
    p = pathlib.Path(path)
    try:
        stat = p.stat()
    except Exception:
        return {
            "name": p.name,
            "size": -1,
            "sha256": "missing",
        }
    cache_key = (
        os.path.normcase(str(p.resolve())),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )
    cached = _FILE_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    digest = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except Exception:
        fingerprint = {
            "name": p.name,
            "size": int(stat.st_size),
            "sha256": "unreadable",
        }
    else:
        fingerprint = {
            "name": p.name,
            "size": int(stat.st_size),
            "sha256": digest.hexdigest(),
        }
    _FILE_FINGERPRINT_CACHE[cache_key] = dict(fingerprint)
    return fingerprint


def _fingerprint_matches(actual, expected) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    for key in ("name", "size", "sha256"):
        if actual.get(key) != expected.get(key):
            return False
    return True


def tensorrt_source_checkpoint_validation_error(
    source_path: str,
    runtime_checkpoint_path: str | None = None,
) -> str | None:
    """Return why a TensorRT source checkpoint is stale/incompatible."""
    source = os.path.abspath(os.path.expanduser(str(source_path)))
    if not os.path.isfile(source):
        return "missing"
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        return f"unreadable: {exc}"
    if not isinstance(checkpoint, dict):
        return "not a checkpoint dictionary"
    if checkpoint.get("checkpoint_format") != _PORTABLE_INT8_CHECKPOINT_FORMAT:
        return "not a portable checkpoint"
    if checkpoint.get("state_format") != _PORTABLE_INT8_STATE_FORMAT:
        return "not a native-FP32 portable checkpoint"
    if checkpoint.get("target_backend") != "tensorrt":
        return "target_backend is not tensorrt"
    if checkpoint.get("tensorrt_source_checkpoint") is not True:
        return "not marked as a TensorRT source checkpoint"
    if checkpoint.get("tensorrt_source_schema") != _TENSORRT_SOURCE_CHECKPOINT_SCHEMA:
        return "TensorRT source schema mismatch"
    expected_signature = _tensorrt_source_signature()
    if checkpoint.get("tensorrt_source_signature") != expected_signature:
        return "TensorRT source signature mismatch"
    if checkpoint.get("activation_quant") != checkpoint.get("source_activation_quant"):
        return "activation qparams do not preserve source checkpoint policy"
    if checkpoint.get("tensorrt_source_activation_quant_policy") != "source":
        return "TensorRT source activation policy mismatch"
    if runtime_checkpoint_path:
        expected = _hash_file_raw(runtime_checkpoint_path)
        actual = checkpoint.get("tensorrt_source_runtime_checkpoint")
        if not _fingerprint_matches(actual, expected):
            return "runtime checkpoint fingerprint mismatch"
    return None


def _regenerate_tensorrt_source_checkpoint(
    runtime_checkpoint_path: str,
    source_path: str,
) -> None:
    quantize_dir = pathlib.Path(_ROOT) / "scripts" / "quantize"
    if not quantize_dir.is_dir():
        raise FileNotFoundError(f"TensorRT source generator missing: {quantize_dir}")
    inserted = False
    quantize_dir_text = str(quantize_dir)
    if quantize_dir_text not in sys.path:
        sys.path.insert(0, quantize_dir_text)
        inserted = True
    try:
        from make_portable_int8_checkpoint import convert_checkpoint

        convert_checkpoint(
            pathlib.Path(runtime_checkpoint_path),
            pathlib.Path(source_path),
            activation_quant="source",
            target_backend="tensorrt",
        )
    finally:
        if inserted:
            try:
                sys.path.remove(quantize_dir_text)
            except ValueError:
                pass
        _FILE_FINGERPRINT_CACHE.clear()


def tensorrt_source_checkpoint_path(model_path: str) -> str:
    """Resolve legacy native-INT8 checkpoints to TensorRT source inputs.

    The current NVIDIA INT8 path uses ModelOpt/QDQ directly from the selected
    PTQ/QAT/QAT-Film checkpoint, so it must not detour through generated
    ``weights/distilled`` files. Source checkpoints remain available
    only for the older native implicit path when ModelOpt INT8 is disabled.
    """
    if not model_path:
        return model_path
    try:
        path = os.path.abspath(os.path.expanduser(str(model_path)))
        if not _is_int8_checkpoint_path(path):
            return path
        if _env_bool("HDRTVNET_TRT_INT8_MODELOPT", _IS_NVIDIA):
            return path
        if _is_tensorrt_source_path(path):
            reason = tensorrt_source_checkpoint_validation_error(path)
            if reason:
                raise RuntimeError(
                    "TensorRT source checkpoint is stale or incompatible: "
                    f"{path} ({reason}). Regenerate from the tracked INT8 "
                    "checkpoint with scripts\\quantize\\make_tensorrt_source_checkpoints.py."
                )
            return path
        candidate = _tensorrt_source_candidate_path(path)
        reason = tensorrt_source_checkpoint_validation_error(candidate, path)
        if reason is None:
            return candidate
        if os.path.isfile(path):
            print(
                "TensorRT source checkpoint refresh: "
                f"{os.path.basename(candidate)} ({reason})"
            )
            _regenerate_tensorrt_source_checkpoint(path, candidate)
            reason = tensorrt_source_checkpoint_validation_error(candidate, path)
            if reason is None:
                return candidate
            raise RuntimeError(
                "Regenerated TensorRT source checkpoint is still invalid: "
                f"{candidate} ({reason})"
            )
        return path
    except RuntimeError:
        raise
    except Exception:
        return model_path


def _unwrap_source_checkpoint(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload.get("state_dict") or {}
        arch = payload.get("architecture") or {}
        if isinstance(arch, dict):
            return state, arch
        return state, {}
    return payload, {}


def _source_checkpoint_architecture(model_path: str) -> dict:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        return {}
    if isinstance(payload, dict):
        arch = payload.get("architecture") or {}
        if isinstance(arch, dict):
            return arch
    return {}


class HDRTVNetTorch:
    """PyTorch inference wrapper for HDRTVNet with platform-aware optimizations.

    Optimizations applied automatically per platform:
      * torch.inference_mode() (lower overhead than no_grad).
      * Pre-allocated GPU tensors to avoid per-frame allocation.
      * CUDA PyTorch path: cudnn.benchmark + optional channels_last tensors.
      * ROCm PyTorch path: contiguous tensors by default for max-autotune.
      * AMD PyTorch path: torch.compile with max-autotune when available.
      * Optional CUDA-graph replay for static-shape inputs.
      * Pre-dequantize INT8 weights on PyTorch backends without tensor cores (auto).
      * Automatic warmup to initialize kernels and eliminate first-frame lag.

    Args:
        warmup_passes: Number of dummy inference passes to run during init.
                       Default=3. Set to 0 to disable. Higher values help
                       torch.compile converge on optimal kernel selection.
    """

    def __init__(self, model_path, device="auto", precision="auto",
                 compile_model=True, force_compile=False, compile_mode="auto",
                 use_cuda_graphs=False, force_channels_last=False,
                 predequantize="auto", hg_weights=None, use_hg=True,
                 warmup_passes=3, fast_condition_resize=False):
        self.model_path = model_path
        self._warmup_passes = warmup_passes
        self._fast_condition_resize = (
            bool(fast_condition_resize)
            or _env_bool("HDRTVNET_FAST_COND_RESIZE", False)
        )
        self._fast_zero_condition = _env_bool("HDRTVNET_ZERO_COND", False)
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
        self._predequantize = _normalize_predequantize_setting(predequantize)
        self._compile_mode = None
        self._assume_aligned_shapes = None
        self._assume_aligned_log_key = None

        # ---- Print platform info ------------------------------------------
        if self._use_cuda:
            gpu_name = _get_gpu_info()
            backend = "ROCm" if _IS_ROCM else "CUDA"
            print(f"GPU: {gpu_name} ({backend})")

        self.model = self._load_model(model_path)

        # ---- Platform-specific: memory format + cudnn.benchmark -----------
        # NVIDIA PyTorch keeps the historical channels_last fast path. ROCm
        # defaults to contiguous because max-autotune can pick worse kernels
        # with channels_last on some AMD cards; opt in with env/CLI to compare.
        self._memory_format_name = "contiguous"
        use_channels_last = (
            self._use_cuda
            and (
                bool(force_channels_last)
                or _IS_NVIDIA
                or (_IS_ROCM and _use_channels_last_by_default())
            )
        )
        if use_channels_last:
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            self.model = self.model.to(memory_format=torch.channels_last)
            self._use_channels_last = True
            self._memory_format_name = "channels-last"
            if _IS_ROCM:
                print("ROCm: cudnn.benchmark + channels_last enabled")
            else:
                print("NVIDIA: cudnn.benchmark + channels_last enabled")
        else:
            self._use_channels_last = False
            if self._use_cuda and _IS_ROCM:
                print("ROCm: channels_last disabled; using contiguous tensors")

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
                        compile_mode = _DEFAULT_TORCH_COMPILE_MODE
                    dynamic = _torch_compile_dynamic_setting()
                    fullgraph = _env_bool("HDRTVNET_COMPILE_FULLGRAPH", False)
                    self.model = torch.compile(
                        self.model,
                        mode=compile_mode,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                    )
                    self._compiled = True
                    self._compile_mode = str(compile_mode)
                    print(
                        "torch.compile enabled "
                        f"(mode={compile_mode}, dynamic={dynamic}, fullgraph={fullgraph})"
                    )
                except Exception as exc:
                    print(f"torch.compile setup failed: {exc}")

        self.expected_hw = None
        self.is_static_input_model = False

        # ---- Pre-allocated buffer state ------------------------------------
        self._buf_hw = None
        self._gpu_input = None       # persistent GPU tensor (1,3,H,W)
        self._gpu_cond = None        # persistent GPU tensor (1,3,H//4,W//4)
        self._gpu_raw = None         # persistent GPU tensor (H,W,3) uint8
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
        if self._fast_condition_resize:
            print("Condition resize: fast bilinear path enabled")

        # ---- Automatic warmup for optimal performance -----------------------
        # Run a few forward passes to initialize kernels, compile shaders,
        # and activate optimization hooks. This eliminates first-frame lag.
        if self._use_cuda and self._warmup_passes > 0:
            print(f"Warming up with {self._warmup_passes} inference passes...")
            self._warmup()

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
    # Warmup
    # -----------------------------------------------------------------------
    def _warmup(self):
        """Run dummy forward passes to warm up kernels and compile caches."""
        if self.model is None:
            return
        # Use a default size if static input size not known yet
        h, w = self.expected_hw or (1080, 1920)
        self._configure_assume_aligned_shapes(w, h)
        cond_h, cond_w = max(1, h // 4), max(1, w // 4)
        mem_fmt = (
            torch.channels_last
            if self._use_channels_last
            else torch.contiguous_format
        )
        dummy = torch.randn(
            (1, 3, h, w),
            device=self.device,
            dtype=self._dtype,
        ).to(memory_format=mem_fmt)
        dummy_cond = torch.randn(
            (1, 3, cond_h, cond_w),
            device=self.device,
            dtype=self._dtype,
        ).to(memory_format=mem_fmt)

        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(self._warmup_passes):
                if self._is_flat_model:
                    _ = self.model(dummy, dummy_cond)
                else:
                    _ = self.model((dummy, dummy_cond))
        if self._use_cuda:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0
        per_pass = elapsed / max(self._warmup_passes, 1) * 1000
        print(f"  Warmup done: {self._warmup_passes} passes in {elapsed:.2f}s "
              f"({per_pass:.1f} ms/pass)")

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
        portable_checkpoint = _is_portable_int8_checkpoint(checkpoint)
        self._int8_quant_type = str(quant_type)
        self._int8_w8a8_layers = set(checkpoint.get("w8a8_layers") or ())
        self._int8_fp16_layers = set(checkpoint.get("fp16_layers") or ())
        try:
            sd_numel = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
        except Exception:
            sd_numel = 0

        if compute_dtype == torch.float32 and self.device.type == "cuda":
            print(
                "WARNING: INT8 checkpoint uses FP32 compute_dtype; this is slower"
                " on GPU. Consider re-quantizing with FP16 compute."
            )

        classifier_default = (
            str(os.environ.get("HDRTVNET_CLASSIFIER", "color_condition")).strip()
            or "color_condition"
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
                classifier=arch.get("classifier", classifier_default),
                cond_c=arch.get("cond_c", 6),
                in_nc=arch.get("in_nc", 3),
                out_nc=arch.get("out_nc", 3),
                nf=arch.get("nf", 32),
                act_type=arch.get("act_type", "relu"),
                weighting_network=arch.get("weighting_network", False),
                hg_nf=arch.get("hg_nf", 64),
                mask_r=arch.get("mask_r", 0.75),
                hg_arch=arch.get("hg_arch", None),
                le_arch=arch.get("le_arch", None),
                post_correction=arch.get("post_correction", None),
            )
        else:
            model = Ensemble_AGCM_LE(
                classifier=arch.get("classifier", classifier_default),
                cond_c=arch.get("cond_c", 6),
                in_nc=arch.get("in_nc", 3),
                out_nc=arch.get("out_nc", 3),
                nf=arch.get("nf", 32),
                act_type=arch.get("act_type", "relu"),
                weighting_network=arch.get("weighting_network", False),
                le_arch=arch.get("le_arch", None),
                post_correction=arch.get("post_correction", None),
            )
        split_hg_requested = bool(self._use_hg and not use_hg)
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

        if portable_checkpoint:
            model.load_state_dict(state_dict, strict=True)

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

        if portable_checkpoint:
            _apply_portable_weight_qparams(model, checkpoint)
            _apply_portable_activation_qparams(model, checkpoint)
            print("Loaded portable INT8 checkpoint base (native FP state + quant metadata)")
        else:
            # Load the quantized wrapper state_dict from legacy runtime checkpoints.
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
            dtype_name = "FP16" if compute_dtype == torch.float16 else "FP32"
            print(f"Loaded {label} INT8 checkpoint ' pre-dequantized to {dtype_name} "
                  f"(compressed storage, native {dtype_name} speed)")
        else:
            self._is_w8_model = True
            label = {"w8a8_full": "W8A8", "w8a8_mixed": "Mixed W8A8/W8A16"}.get(
                quant_type, quant_type)
            if quant_type == "w8a8_mixed" and checkpoint.get("fp16_layers"):
                label = "Mixed W8A8/W8A16/FP16"
            print(f"Loaded {label} INT8 model (compute_dtype={compute_dtype})")
        if split_hg_requested:
            hg_weights_path, searched_hg_paths = self._resolve_hg_weights(model_path)
            if not hg_weights_path:
                searched = "\n".join(f"  - {p}" for p in searched_hg_paths)
                raise FileNotFoundError(
                    "INT8 HG weights were requested but not found.\n"
                    "  Searched paths:\n"
                    f"{searched}\n"
                    "  Pass --hg-weights or disable HG."
                )
            hg_module = self._load_split_int8_hg_module(
                hg_weights_path,
                compute_dtype=compute_dtype,
                predequantize=bool(should_predequantize and should_predequantize != "auto"),
            )
            composite = HG_Composite(
                classifier=arch.get("classifier", classifier_default),
                cond_c=arch.get("cond_c", 6),
                in_nc=arch.get("in_nc", 3),
                out_nc=arch.get("out_nc", 3),
                nf=arch.get("nf", 32),
                act_type=arch.get("act_type", "relu"),
                weighting_network=arch.get("weighting_network", False),
                hg_nf=arch.get("hg_nf", 64),
                mask_r=arch.get("mask_r", 0.75),
                hg_arch=(torch.load(hg_weights_path, map_location="cpu", weights_only=False).get("architecture", {}) or {}).get("hg_arch"),
                le_arch=arch.get("le_arch", None),
                post_correction=arch.get("post_correction", None),
            ).to(dtype=compute_dtype, device=self.device)
            composite.base = model
            composite.hg = hg_module
            composite.eval()
            model = composite
            self._use_hg = True
            print(f"Attached split INT8 HG weights: {hg_weights_path}")
        return model

    def _load_split_int8_hg_module(self, hg_weights_path, *, compute_dtype, predequantize: bool):
        checkpoint = torch.load(hg_weights_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"{hg_weights_path} is not an INT8 HG checkpoint.")
        arch = checkpoint.get("architecture", {}) or {}
        holder = HG_Composite(
            classifier="color_condition",
            cond_c=6,
            in_nc=arch.get("in_nc", 3),
            out_nc=arch.get("out_nc", 3),
            nf=32,
            act_type="relu",
            weighting_network=False,
            hg_nf=arch.get("hg_nf", 64),
            mask_r=arch.get("mask_r", 0.75),
            hg_arch=arch.get("hg_arch", None),
            le_arch=None,
        )
        hg = holder.hg
        quant_type = checkpoint.get("quantization", "w8_weight_only")
        state_dict = checkpoint.get("state_dict", {})
        portable_checkpoint = _is_portable_int8_checkpoint(checkpoint)
        if portable_checkpoint:
            hg.load_state_dict(state_dict, strict=True)

        if quant_type == "w8a8_mixed":
            use_asym = checkpoint.get("activation_quant", "symmetric") == "asymmetric"
            _quantize_model_mixed_v2(
                hg,
                compute_dtype,
                w8a8_layers=checkpoint.get("w8a8_layers", None),
                fp16_layers=checkpoint.get("fp16_layers", None),
                asymmetric=use_asym,
            )
        elif quant_type == "w8a8_full":
            use_asym = checkpoint.get("activation_quant", "symmetric") == "asymmetric"
            _quantize_model_w8a8(hg, compute_dtype, asymmetric=use_asym)
        else:
            raise ValueError(f"Unknown HG quantization type '{quant_type}'")

        if portable_checkpoint:
            _apply_portable_weight_qparams(hg, checkpoint)
            _apply_portable_activation_qparams(hg, checkpoint)
        else:
            hg.load_state_dict(state_dict, strict=True)
        hg = hg.to(dtype=compute_dtype, device=self.device)
        if predequantize:
            _predequantize_model(hg, compute_dtype)
        hg.eval()
        return hg

    def _resolve_hg_weights(self, model_path):
        """Find original or distilled HG weights from common locations."""
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
        _add(os.path.join(os.path.dirname(os.path.abspath(model_path)), "HG.pt"))
        _add(
            os.path.join(
                os.path.dirname(os.path.abspath(model_path)),
                "HG_qfriendly_directh16_fp32.pt",
            )
        )
        # Repo-default runtime path.
        _add(_DEFAULT_HG_WEIGHTS)
        # Relative path from current working directory (when launched outside repo root).
        _add(os.path.join(os.getcwd(), "src", "models", "weights", "original", "HG.pt"))

        for path in candidates:
            if os.path.isfile(path):
                return path, candidates
        return None, candidates

    def _load_model(self, model_path):
        # INT8 quantized models use a different loading path
        if self.precision in ("int8-full", "int8-mixed"):
            return self._load_int8_model(model_path)

        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".ts":
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
        else:
            use_hg = bool(self._use_hg)
            hg_weights_path = None
            searched_hg_paths = []
            base_payload = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
            base_state, base_arch = _unwrap_source_checkpoint(base_payload)
            if not isinstance(base_arch, dict):
                base_arch = {}
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
                        "  To enable HG, place original/HG.pt under "
                        "src/models/weights/ or pass --hg-weights."
                    )
                    use_hg = False

            self._use_hg = use_hg
            classifier_name = (
                str(
                    base_arch.get(
                        "classifier",
                        os.environ.get("HDRTVNET_CLASSIFIER", "color_condition"),
                    )
                ).strip()
                or "color_condition"
            )
            le_arch = (
                str(base_arch.get("le_arch", os.environ.get("HDRTVNET_LE_ARCH", ""))).strip()
                or None
            )
            hg_arch = (
                str(base_arch.get("hg_arch", os.environ.get("HDRTVNET_HG_ARCH", ""))).strip()
                or None
            )
            post_correction = (
                str(
                    base_arch.get(
                        "post_correction",
                        os.environ.get("HDRTVNET_POST_CORRECTION", ""),
                    )
                ).strip()
                or None
            )

            if use_hg:
                model = HG_Composite(
                    classifier=classifier_name,
                    cond_c=6,
                    in_nc=3,
                    out_nc=3,
                    nf=32,
                    act_type="relu",
                    weighting_network=False,
                    hg_nf=64,
                    mask_r=0.75,
                    hg_arch=hg_arch,
                    le_arch=le_arch,
                    post_correction=post_correction,
                ).to(self.device)

                cleaned = {}
                for key, value in base_state.items():
                    cleaned[key[7:] if key.startswith("module.") else key] = value
                model.base.load_state_dict(cleaned, strict=True)

                hg_state = torch.load(hg_weights_path, map_location=self.device)
                hg_state, _hg_arch = _unwrap_source_checkpoint(hg_state)
                model.hg.load_state_dict(hg_state, strict=True)
                model.eval()
            else:
                model = Ensemble_AGCM_LE(
                    classifier=classifier_name,
                    cond_c=6,
                    in_nc=3,
                    out_nc=3,
                    nf=32,
                    act_type="relu",
                    weighting_network=False,
                    le_arch=le_arch,
                    post_correction=post_correction,
                ).to(self.device)

                cleaned = {}
                for key, value in base_state.items():
                    cleaned[key[7:] if key.startswith("module.") else key] = value
                model.load_state_dict(cleaned, strict=True)
                model.eval()
            setattr(model, "_hdrtvnet_classifier_arch", classifier_name)
            setattr(model, "_hdrtvnet_le_arch", le_arch or "")
            setattr(model, "_hdrtvnet_hg_arch", hg_arch or "")
            setattr(model, "_hdrtvnet_post_correction", post_correction or "")

        if self.precision == "fp16" and self.device.type == "cuda":
            model = model.half()
        else:
            model = model.float()

        return model

    def _configure_assume_aligned_shapes(self, width: int, height: int) -> None:
        enabled = _assume_aligned_shapes_for_resolution(width, height)
        root = getattr(self.model, "_orig_mod", self.model)
        updated = 0
        try:
            modules = root.modules()
        except Exception:
            modules = ()
        for module in modules:
            if hasattr(module, "assume_aligned_shapes"):
                if bool(getattr(module, "assume_aligned_shapes")) != enabled:
                    setattr(module, "assume_aligned_shapes", enabled)
                    updated += 1

        self._assume_aligned_shapes = enabled
        log_key = (int(width), int(height), bool(enabled))
        if self._assume_aligned_log_key != log_key:
            state = "enabled" if enabled else "disabled"
            print(
                f"Aligned-shape fast graph {state} for "
                f"{int(width)}x{int(height)}"
            )
            self._assume_aligned_log_key = log_key

    # -----------------------------------------------------------------------
    # Buffer management " allocate once, reuse every frame
    # -----------------------------------------------------------------------
    def _ensure_buffers(self, h, w):
        """Allocate / reallocate persistent GPU tensors when resolution changes.
        For the common case (fixed resolution) this is a no-op after the first
        frame."""
        self._configure_assume_aligned_shapes(w, h)
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
            self._gpu_raw = torch.empty(
                (h, w, 3), dtype=torch.uint8, device=self.device
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
            self._gpu_raw.copy_(
                self._pin_input,
                non_blocking=True,                              # pinned'GPU (async DMA)
            )
            raw = self._gpu_raw
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

        # Condition tensor (0.25- spatial). The zero-condition path is an
        # explicit speed-only TensorRT INT8 shortcut; FP16 keeps the normal path.
        if self._fast_zero_condition:
            self._gpu_cond.zero_()
            return self._gpu_input, self._gpu_cond
        elif self._fast_condition_resize:
            cond = F.interpolate(
                self._gpu_input,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        else:
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

        self._configure_assume_aligned_shapes(width, height)
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
            warmup_runs = _torch_compile_warmup_runs()
            for i in range(warmup_runs):
                self.process(dummy)  # triggers full compile -> Triton disk cache
                if not self._compiled:
                    raise RuntimeError(
                        "torch.compile failed during warmup and fell back to "
                        "eager mode; no compiled kernel cache was generated."
                    )
                if self._use_cuda:
                    torch.cuda.synchronize()
                if warmup_runs > 1:
                    print(
                        f"[compile] warmup pass {i + 1}/{warmup_runs} complete",
                        flush=True,
                    )
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


def _tensorrt_calibration_dir(create: bool = False) -> str:
    root = os.environ.get(
        "HDRTVNET_TRT_PREBUILT_CALIBRATION_DIR",
        os.path.join(_ROOT, "src", "models", _TENSORRT_CALIBRATION_SUBDIR),
    )
    if create:
        os.makedirs(root, exist_ok=True)
    return root


def _hash_file(path: str) -> dict[str, object]:
    path = tensorrt_source_checkpoint_path(path)
    return _hash_file_raw(path)


def _tensorrt_source_signature() -> str:
    global _TENSORRT_SOURCE_SIGNATURE_CACHE
    if _TENSORRT_SOURCE_SIGNATURE_CACHE:
        return _TENSORRT_SOURCE_SIGNATURE_CACHE
    root = pathlib.Path(_ROOT)
    digest = hashlib.sha256()
    digest.update(_TENSORRT_SOURCE_SIGNATURE_VERSION.encode("utf-8"))
    digest.update(b"\0")
    models_dir = pathlib.Path(_HERE)
    generated_dirs = {"__pycache__", "compile_cache", "engines", "onnx"}
    files: list[pathlib.Path] = []
    if models_dir.is_dir():
        for p in sorted(models_dir.rglob("*.py")):
            try:
                rel_parts = set(p.relative_to(models_dir).parts)
            except ValueError:
                rel_parts = set(p.parts)
            if rel_parts & generated_dirs:
                continue
            files.append(p)
    for p in files:
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.name
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(p.read_bytes())
        except Exception:
            digest.update(b"<missing>")
        digest.update(b"\0")
    _TENSORRT_SOURCE_SIGNATURE_CACHE = digest.hexdigest()
    return _TENSORRT_SOURCE_SIGNATURE_CACHE


def _safe_tensorrt_version(trt_module=None) -> str:
    try:
        trt_module = trt_module or __import__("tensorrt")
        return str(getattr(trt_module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def _safe_cuda_driver_version() -> str:
    try:
        fn = getattr(torch.cuda, "driver_version", None)
        if callable(fn):
            return str(fn())
    except Exception:
        pass
    try:
        fn = getattr(torch._C, "_cuda_getDriverVersion", None)
        if callable(fn):
            return str(fn())
    except Exception:
        pass
    return "unknown"


def _cuda_device_fingerprint() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        uuid = getattr(props, "uuid", "")
        if isinstance(uuid, bytes):
            uuid = uuid.hex()
        return {
            "available": True,
            "index": int(index),
            "name": str(getattr(props, "name", "")),
            "capability": [
                int(getattr(props, "major", 0)),
                int(getattr(props, "minor", 0)),
            ],
            "total_memory": int(getattr(props, "total_memory", 0)),
            "multi_processor_count": int(
                getattr(props, "multi_processor_count", 0)
            ),
            "uuid": str(uuid or ""),
        }
    except Exception as exc:
        return {"available": True, "error": type(exc).__name__}


def _normalize_tensorrt_qdq_fusion(qdq_fusion: str) -> str:
    fusion = str(qdq_fusion or "auto").strip().lower()
    if fusion in {
        "native",
        "native-int8",
        "native_int8",
        "implicit",
        "implicit-int8",
        "implicit_int8",
        "calibrated",
        "calibration",
        "ptq",
    }:
        return "native"
    if fusion in {"auto", "default", "smart"}:
        return "auto"
    if fusion in {"add", "residual-add", "add-inputs"}:
        return "add"
    if fusion in {
        "add-mul",
        "addmul",
        "elementwise",
        "elementwise-inputs",
        "mul",
        "all",
    }:
        return "add-mul"
    return "none"


def _resolve_tensorrt_qdq_fusion(precision: str, qdq_fusion: str) -> str:
    fusion = _normalize_tensorrt_qdq_fusion(qdq_fusion)
    if fusion == "native":
        return "native"
    if fusion != "auto":
        return fusion
    if str(precision or "").startswith("int8"):
        return "native"
    return "none"


def tensorrt_engine_metadata_path(engine_path: str) -> str:
    return f"{engine_path}.json"


def _tensorrt_expected_engine_metadata(
    *,
    model_path: str,
    width: int,
    height: int,
    precision: str,
    mode_name: str,
    engine_mode_name: str,
    builder_precision: str,
    use_hg: bool,
    predequantize_int8: bool,
    qdq_fusion: str,
    hg_weights: str | None = None,
    calibration_dataset: str | None = None,
    calibration_video: str | None = None,
    calibration_frames: int | None = None,
    calibration_cache: str | None = None,
    trt_module=None,
) -> dict[str, object]:
    model_path = tensorrt_source_checkpoint_path(model_path)
    hg_path = hg_weights or _DEFAULT_HG_WEIGHTS
    aux_streams = _tensorrt_aux_stream_count()
    resolved_qdq_fusion = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
    fp16_builder_enabled = _tensorrt_fp16_enabled()
    condition_free_single_input = _tensorrt_condition_free_arch_enabled_for_checkpoint(
        model_path
    )
    effective_modelopt_torch = _tensorrt_int8_modelopt_torch_effective_enabled(
        mode_name
    )
    if str(precision).startswith("int8-full") and (
        not _tensorrt_full_int8_fp16_islands_enabled()
        and not (
            effective_modelopt_torch
            and not _tensorrt_int8_modelopt_strongly_typed()
        )
    ):
        fp16_builder_enabled = False

    metadata = {
        "schema": _TENSORRT_ENGINE_METADATA_SCHEMA,
        "source_signature": _tensorrt_source_signature(),
        "model": _hash_file(model_path),
        "hg": {
            "enabled": bool(use_hg),
            "weights": _hash_file(hg_path) if use_hg else None,
        },
        "engine": {
            "width": int(width),
            "height": int(height),
            "precision": str(precision),
            "builder_precision": str(builder_precision),
            "mode_name": str(mode_name),
            "engine_mode_name": str(engine_mode_name),
            "use_hg": bool(use_hg),
            "predequantize_int8": bool(predequantize_int8),
            "qdq_fusion": resolved_qdq_fusion,
            "fp_export_prefixes": (
                list(_tensorrt_int8_fp_export_prefixes())
                if (
                    str(precision).startswith("int8")
                    and resolved_qdq_fusion != "native"
                ) else []
            ),
        },
        "runtime": {
            "torch": str(getattr(torch, "__version__", "unknown")),
            "torch_cuda": str(getattr(torch.version, "cuda", None)),
            "tensorrt": _safe_tensorrt_version(trt_module),
            "cuda_driver": _safe_cuda_driver_version(),
            "device": _cuda_device_fingerprint(),
        },
        "build": {
            "builder_optimization_level": _tensorrt_builder_optimization_level(),
            "fp16_enabled": fp16_builder_enabled,
            "condition_free_single_input": condition_free_single_input,
            "workspace_gb": _tensorrt_workspace_gb(),
            "aux_streams": aux_streams,
        },
    }
    if str(precision).startswith("int8"):
        qat_composition = _tensorrt_int8_qat_composition_policy(mode_name)
        qat_checkpoint_composition = (
            _tensorrt_int8_qat_checkpoint_composition_enabled(mode_name)
        )
        metadata["build"]["int8_modelopt"] = _tensorrt_int8_modelopt_enabled()
        metadata["build"]["int8_modelopt_torch"] = effective_modelopt_torch
        metadata["build"]["int8_modelopt_torch_requested"] = (
            _tensorrt_int8_modelopt_torch_enabled()
        )
        if _tensorrt_mode_name_is_qat(mode_name):
            metadata["build"]["int8_qat_composition"] = qat_composition
            metadata["build"]["int8_qat_checkpoint_composition"] = (
                qat_checkpoint_composition
            )
        metadata["build"]["int8_modelopt_torch_mode"] = (
            _tensorrt_int8_modelopt_torch_mode(precision)
        )
        metadata["build"]["int8_modelopt_torch_method"] = (
            _tensorrt_int8_modelopt_torch_method()
        )
        metadata["build"]["int8_modelopt_torch_effective_bits"] = (
            _tensorrt_int8_modelopt_torch_effective_bits()
        )
        metadata["build"]["int8_modelopt_torch_calib_steps"] = (
            _tensorrt_int8_modelopt_torch_calib_steps()
        )
        metadata["build"]["int8_modelopt_torch_score_steps"] = (
            _tensorrt_int8_modelopt_torch_score_steps()
        )
        metadata["build"]["int8_modelopt_torch_include"] = list(
            _tensorrt_int8_modelopt_torch_include_patterns()
        )
        metadata["build"]["int8_modelopt_torch_exclude"] = list(
            _tensorrt_int8_modelopt_torch_exclude_patterns()
        )
        metadata["build"]["int8_modelopt_torch_qdq_dtype"] = (
            _tensorrt_int8_modelopt_torch_qdq_dtype(torch.float16)
        )
        metadata["build"]["int8_modelopt_torch_quant_scheme"] = (
            _tensorrt_int8_modelopt_torch_quant_scheme()
        )
        metadata["build"]["int8_modelopt_torch_hg_default"] = (
            _tensorrt_int8_modelopt_torch_hg_default_enabled()
        )
        metadata["build"]["int8_modelopt_torch_hg_mixed_exclude"] = list(
            _tensorrt_int8_modelopt_torch_hg_mixed_exclude_patterns()
        )
        metadata["build"]["int8_modelopt_ops"] = _tensorrt_int8_modelopt_ops()
        metadata["build"]["int8_modelopt_calibration"] = (
            _tensorrt_int8_modelopt_calibration_method()
        )
        metadata["build"]["int8_modelopt_layer_policy"] = (
            _tensorrt_int8_modelopt_layer_policy()
        )
        metadata["build"]["int8_modelopt_node_override"] = (
            _tensorrt_int8_modelopt_node_override()
        )
        metadata["build"]["int8_modelopt_autotune"] = (
            _tensorrt_int8_modelopt_autotune_enabled()
        )
        metadata["build"]["int8_modelopt_autotune_filter"] = (
            _tensorrt_int8_modelopt_autotune_filter()
        )
        metadata["build"]["int8_modelopt_autotune_schemes"] = (
            _tensorrt_int8_modelopt_autotune_int(
                "HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_SCHEMES",
                12,
                1,
            )
        )
        metadata["build"]["int8_modelopt_strongly_typed"] = (
            _tensorrt_int8_modelopt_strongly_typed()
        )
        metadata["build"]["int8_modelopt_auto_profile"] = (
            _tensorrt_int8_modelopt_auto_profile_enabled()
        )
        try:
            metadata_onnx_path = tensorrt_onnx_path(
                model_path,
                width,
                height,
                engine_mode_name,
            )
            metadata["build"]["int8_modelopt_prequantized_onnx"] = (
                _tensorrt_int8_modelopt_prequantized_onnx(metadata_onnx_path)
            )
        except Exception:
            metadata["build"]["int8_modelopt_prequantized_onnx"] = (
                _tensorrt_int8_modelopt_prequantized_onnx(None)
            )
    if resolved_qdq_fusion == "native" and str(precision).startswith("int8"):
        metadata["build"]["native_int8_prefer_constraints"] = (
            _tensorrt_native_int8_prefer_constraints()
        )
        metadata["build"]["native_int8_obey_constraints"] = (
            _tensorrt_native_int8_obey_constraints()
        )
        metadata["build"]["native_int8_checkpoint_policy"] = (
            _tensorrt_native_int8_checkpoint_policy_enabled()
        )
        metadata["build"]["native_int8_policy_obey"] = (
            _tensorrt_native_int8_policy_obey()
        )
        metadata["build"]["native_int8_policy_int8_outputs"] = (
            _tensorrt_native_int8_policy_int8_outputs()
        )
        metadata["build"]["native_int8_policy_output_constraints"] = (
            _tensorrt_native_int8_policy_output_constraints()
        )
    if str(precision).startswith("int8"):
        if predequantize_int8:
            metadata["build"]["int8_zero_condition"] = False
            metadata["build"]["int8_single_input"] = False
        else:
            metadata["build"]["int8_zero_condition"] = _env_bool(
                "HDRTVNET_TRT_INT8_ZERO_COND",
                _tensorrt_int8_zero_condition_enabled(mode_name),
            )
            metadata["build"]["int8_single_input"] = (
                condition_free_single_input
                or _tensorrt_int8_single_input_enabled(mode_name)
            )
        metadata["build"]["int8_agcm_only"] = _env_bool(
            "HDRTVNET_TRT_INT8_AGCM_ONLY",
            _tensorrt_int8_agcm_only_enabled(mode_name),
        )
    if str(precision).startswith("int8-full"):
        metadata["build"]["full_int8_fp16_islands"] = (
            _tensorrt_full_int8_fp16_islands_enabled()
        )
    calibration = _tensorrt_calibration_fingerprint(
        precision=precision,
        qdq_fusion=resolved_qdq_fusion,
        calibration_dataset=calibration_dataset,
        calibration_video=calibration_video,
        calibration_frames=calibration_frames,
        calibration_cache=calibration_cache,
    )
    if calibration is not None:
        metadata["calibration"] = calibration
    return metadata


def _first_metadata_mismatch(expected, actual, prefix: str = "") -> str | None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{prefix or 'metadata'} type mismatch"
        for key, expected_value in expected.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in actual:
                return f"{child_prefix} missing"
            mismatch = _first_metadata_mismatch(
                expected_value,
                actual.get(key),
                child_prefix,
            )
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if list(actual or []) != expected:
            return f"{prefix} expected {expected!r}, got {actual!r}"
        return None
    if actual != expected:
        return f"{prefix} expected {expected!r}, got {actual!r}"
    return None


def _tensorrt_engine_validation_error(
    engine_path: str,
    expected: dict[str, object],
) -> str | None:
    if not engine_path or not os.path.isfile(engine_path):
        return "engine file missing"
    meta_path = tensorrt_engine_metadata_path(engine_path)
    if not os.path.isfile(meta_path):
        return "metadata sidecar missing"
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            actual = json.load(fh)
    except Exception as exc:
        return f"metadata unreadable: {exc}"
    return _first_metadata_mismatch(expected, actual)


def tensorrt_engine_is_valid(
    engine_path: str,
    *,
    model_path: str,
    width: int,
    height: int,
    precision: str,
    mode_name: str,
    use_hg: bool,
    predequantize=False,
    qdq_fusion: str = "auto",
    hg_weights: str | None = None,
    calibration_dataset: str | None = None,
    calibration_video: str | None = None,
    calibration_frames: int | None = None,
    calibration_cache: str | None = None,
    verbose: bool = False,
) -> bool:
    predeq = _resolve_tensorrt_predequantize(precision, predequantize)
    qdq_fusion = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
    calibration_dataset, calibration_video, calibration_cache, _ = (
        _resolve_tensorrt_calibration_sources(
            model_path=model_path,
            width=width,
            height=height,
            precision=precision,
            mode_name=mode_name,
            use_hg=use_hg,
            predequantize=predeq,
            qdq_fusion=qdq_fusion,
            calibration_dataset=calibration_dataset,
            calibration_video=calibration_video,
            calibration_cache=calibration_cache,
        )
    )
    engine_mode_name = tensorrt_mode_name(
        precision,
        mode_name,
        predequantize=predeq,
        qdq_fusion=qdq_fusion,
    )
    builder_precision = "fp16" if predeq else str(precision)
    expected = _tensorrt_expected_engine_metadata(
        model_path=model_path,
        width=width,
        height=height,
        precision=precision,
        mode_name=mode_name,
        engine_mode_name=engine_mode_name,
        builder_precision=builder_precision,
        use_hg=use_hg,
        predequantize_int8=predeq,
        qdq_fusion=qdq_fusion,
        hg_weights=hg_weights,
        calibration_dataset=calibration_dataset,
        calibration_video=calibration_video,
        calibration_frames=calibration_frames,
        calibration_cache=calibration_cache,
    )
    reason = _tensorrt_engine_validation_error(engine_path, expected)
    if reason and verbose:
        print(f"TensorRT engine cache invalid: {reason} ({engine_path})")
    return reason is None


def _write_tensorrt_engine_metadata(
    engine_path: str,
    metadata: dict[str, object],
) -> None:
    meta_path = tensorrt_engine_metadata_path(engine_path)
    payload = dict(metadata)
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        pathlib.Path(meta_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"TensorRT engine metadata saved: {meta_path}")
    except Exception as exc:
        print(f"TensorRT engine metadata save skipped: {exc}")


def _tensorrt_timing_cache_path():
    override = os.environ.get("HDRTVNET_TRT_TIMING_CACHE")
    if override is not None:
        text = str(override).strip()
        if text.lower() in {"", "0", "false", "off", "none", "disable", "disabled"}:
            return None
        return os.path.abspath(os.path.expanduser(text))
    return os.path.join(_engine_cache_dir(), "tensorrt_timing.cache")


def _tensorrt_builder_optimization_level() -> int:
    try:
        value = int(os.environ.get("HDRTVNET_TRT_BUILDER_OPT_LEVEL", "5"))
    except Exception:
        value = 5
    return min(5, max(0, value))


def _tensorrt_aux_stream_count():
    value = os.environ.get("HDRTVNET_TRT_AUX_STREAMS")
    if value is None or str(value).strip() == "":
        return None
    try:
        return max(0, int(value))
    except Exception:
        return None


def _tensorrt_workspace_gb() -> float | None:
    value = os.environ.get("HDRTVNET_TRT_WORKSPACE_GB")
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in {"0", "none", "unlimited", "default", "auto"}:
        return None
    try:
        parsed = float(text)
    except Exception:
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _tensorrt_fp16_enabled() -> bool:
    return _env_bool("HDRTVNET_TRT_FP16", True)


def _tensorrt_int8_modelopt_enabled() -> bool:
    return _env_bool("HDRTVNET_TRT_INT8_MODELOPT", _IS_NVIDIA)


def _tensorrt_int8_modelopt_torch_enabled() -> bool:
    return _env_bool("HDRTVNET_TRT_INT8_MODELOPT_TORCH", _IS_NVIDIA)


def _tensorrt_mode_name_is_qat(mode_name: str | None = None) -> bool:
    text = str(mode_name or "").strip().lower()
    return "qat" in text


def _tensorrt_int8_qat_composition_policy(mode_name: str | None = None) -> str:
    if not _tensorrt_mode_name_is_qat(mode_name):
        return "runtime"
    text = str(
        os.environ.get("HDRTVNET_TRT_INT8_QAT_COMPOSITION", "runtime")
    ).strip().lower()
    if text in {"torch", "runtime", "search", "modelopt-torch", "tuned"}:
        return "runtime"
    return "checkpoint"


def _tensorrt_int8_qat_checkpoint_composition_enabled(
    mode_name: str | None = None,
) -> bool:
    return _tensorrt_int8_qat_composition_policy(mode_name) == "checkpoint"


def _tensorrt_int8_modelopt_torch_effective_enabled(
    mode_name: str | None = None,
) -> bool:
    return (
        _tensorrt_int8_modelopt_torch_enabled()
        and not _tensorrt_int8_qat_checkpoint_composition_enabled(mode_name)
    )


def _tensorrt_int8_modelopt_torch_mode(precision: str | None = None) -> str:
    text = str(os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_TORCH_MODE", "")).strip().lower()
    if text in {"full", "all", "int8"}:
        return "full"
    if text in {"auto", "mixed", "balance", "balanced", "search"}:
        return "auto"
    return "full" if str(precision or "").startswith("int8-full") else "auto"


def _tensorrt_int8_modelopt_torch_method() -> str:
    text = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_TORCH_METHOD", "gradient")
    ).strip().lower()
    return text if text in {"gradient", "kl_div"} else "gradient"


def _tensorrt_int8_modelopt_torch_int(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)
    return max(int(minimum), value)


def _tensorrt_int8_modelopt_torch_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def _tensorrt_int8_modelopt_torch_calib_steps() -> int:
    return _tensorrt_int8_modelopt_torch_int(
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_CALIB_STEPS",
        8,
        1,
    )


def _tensorrt_int8_modelopt_torch_score_steps() -> int:
    return _tensorrt_int8_modelopt_torch_int(
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_SCORE_STEPS",
        4,
        1,
    )


def _tensorrt_int8_modelopt_torch_effective_bits() -> float:
    return _tensorrt_int8_modelopt_torch_float(
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_EFFECTIVE_BITS",
        10.0,
        8.0,
        16.0,
    )


def _tensorrt_int8_modelopt_torch_seed() -> int:
    return _tensorrt_int8_modelopt_torch_int(
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_SEED",
        1234,
        0,
    )


def _tensorrt_int8_modelopt_torch_patterns(name: str) -> tuple[str, ...]:
    text = str(os.environ.get(name, "") or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in re.split(r"[;,]", text) if part.strip())


def _tensorrt_int8_modelopt_torch_include_patterns() -> tuple[str, ...]:
    explicit = _tensorrt_int8_modelopt_torch_patterns(
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_INCLUDE"
    )
    if explicit:
        return explicit
    return (
        "base.AGCM.spatial",
        "base.AGCM.global",
        "base.LE.low_in",
        "base.LE.recon_trunk3",
        "base.post_correction.trunk",
        "base.post_correction.out",
        "base.post_correction.net",
        "AGCM.spatial",
        "AGCM.global",
        "LE.low_in",
        "LE.recon_trunk3",
        "post_correction.trunk",
        "post_correction.out",
        "post_correction.net",
        "hg.low_in",
        "hg.trunk",
    )


def _tensorrt_int8_modelopt_torch_exclude_patterns() -> tuple[str, ...]:
    return _tensorrt_int8_modelopt_torch_patterns(
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_EXCLUDE"
    )


def _tensorrt_int8_modelopt_torch_hg_default_enabled() -> bool:
    return _env_bool("HDRTVNET_TRT_INT8_MODELOPT_TORCH_HG_DEFAULT", True)


def _tensorrt_int8_modelopt_torch_hg_mixed_exclude_patterns() -> tuple[str, ...]:
    explicit = _tensorrt_int8_modelopt_torch_patterns(
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_HG_MIXED_EXCLUDE"
    )
    if explicit:
        return explicit
    return ("hg\\.conv1", "hg\\.conv_last")


def _tensorrt_int8_modelopt_torch_qdq_dtype(dtype: torch.dtype) -> str:
    text = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_TORCH_QDQ_DTYPE", "auto")
    ).strip().lower()
    if text in {"fp32", "float", "float32"}:
        return "Float"
    if text in {"fp16", "half", "float16"}:
        return "Half"
    return "Half" if dtype == torch.float16 else "Float"


def _tensorrt_int8_modelopt_torch_quant_scheme() -> str:
    text = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_TORCH_QUANT_SCHEME", "symmetric")
    ).strip().lower()
    if text in {"asym", "asymmetric", "unsigned", "uint8"}:
        return "asymmetric"
    return "symmetric"


def _tensorrt_int8_modelopt_strongly_typed() -> bool:
    return _env_bool("HDRTVNET_TRT_INT8_MODELOPT_STRONGLY_TYPED", True)


def _tensorrt_int8_modelopt_auto_profile_enabled() -> bool:
    if _tensorrt_int8_modelopt_torch_enabled():
        return False
    return _env_bool("HDRTVNET_TRT_INT8_MODELOPT_AUTO_PROFILE", True)


def _tensorrt_int8_modelopt_auto_profile_path(onnx_path: str) -> str:
    root, _ext = os.path.splitext(str(onnx_path))
    return os.path.join(
        os.path.dirname(root),
        "modelopt_autotune",
        f"{pathlib.Path(root).name}_modelopt_int8",
        "region_models",
        "region_19_level_0.onnx",
    )


def _tensorrt_int8_modelopt_prequantized_onnx(onnx_path: str | None = None) -> str | None:
    explicit = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_PREQUANTIZED_ONNX", "")
    ).strip()
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    if onnx_path and _tensorrt_int8_modelopt_auto_profile_enabled():
        candidate = os.path.abspath(_tensorrt_int8_modelopt_auto_profile_path(onnx_path))
        if os.path.isfile(candidate):
            return candidate
    return None


def _tensorrt_int8_modelopt_calibration_method() -> str:
    method = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_CALIBRATION", "max")
    ).strip().lower()
    return method if method in {"max", "entropy"} else "max"


def _tensorrt_int8_modelopt_ops() -> list[str] | None:
    text = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_OPS", "Conv,Gemm,MatMul")
    ).strip()
    if not text or text.lower() in {"all", "default", "none", "*"}:
        return None if text.lower() in {"all", "*"} else ["Conv", "Gemm", "MatMul"]
    ops = [
        item.strip()
        for item in re.split(r"[,;\s]+", text)
        if item.strip()
    ]
    return ops or ["Conv", "Gemm", "MatMul"]


def _tensorrt_int8_modelopt_layer_policy() -> str:
    text = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_LAYER_POLICY", "checkpoint")
    ).strip().lower()
    if text in {"all", "ops", "operator", "operators"}:
        return "all"
    return "checkpoint"


def _tensorrt_int8_modelopt_node_override() -> list[str] | None:
    text = str(os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_NODES", "")).strip()
    if not text:
        return None
    if text.startswith("@"):
        path = pathlib.Path(text[1:]).expanduser()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"TensorRT INT8 ModelOpt node override skipped: {exc}")
            return None
    nodes = [
        item.strip()
        for item in re.split(r"[,;\s]+", text)
        if item.strip()
    ]
    return nodes or None


def _tensorrt_int8_modelopt_autotune_enabled() -> bool:
    if _tensorrt_int8_modelopt_torch_enabled():
        return False
    return _env_bool("HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE", False)


def _tensorrt_int8_modelopt_autotune_int(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(minimum), int(value))


def _tensorrt_int8_modelopt_autotune_filter() -> list[str] | None:
    text = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_FILTER", "")
    ).strip()
    if not text:
        return None
    if text.startswith("@"):
        path = pathlib.Path(text[1:]).expanduser()
        try:
            return [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ] or None
        except Exception as exc:
            print(f"TensorRT INT8 ModelOpt autotune filter skipped: {exc}")
            return None
    return [
        item.strip()
        for item in re.split(r"[,;\s]+", text)
        if item.strip()
    ] or None


def _tensorrt_full_int8_fp16_islands_enabled() -> bool:
    return _env_bool("HDRTVNET_TRT_FULL_INT8_FP16_ISLANDS", False)


def _tensorrt_full_int8_safe_fp16_builder_enabled() -> bool:
    return _env_bool("HDRTVNET_TRT_FULL_INT8_SAFE_FP16_BUILDER", True)


def _tensorrt_int8_agcm_only_default(mode_name: str | None = None) -> bool:
    # Keep the TensorRT graph comparable across precisions by default.  The
    # AGCM-only export is a deployment shortcut, not a fair quantization test.
    return False


def _tensorrt_mode_name_is_mixed_int8(mode_name: str | None = None) -> bool:
    text = str(mode_name or "").strip().lower()
    return "mixed" in text and "int8" in text


def _tensorrt_mode_name_is_full_int8(mode_name: str | None = None) -> bool:
    text = str(mode_name or "").strip().lower()
    return "full" in text and "int8" in text


def _tensorrt_int8_agcm_only_enabled(mode_name: str | None = None) -> bool:
    return _env_bool(
        "HDRTVNET_TRT_INT8_AGCM_ONLY",
        _tensorrt_int8_agcm_only_default(mode_name),
    )


def _tensorrt_int8_zero_condition_enabled(mode_name: str | None = None) -> bool:
    return _env_bool(
        "HDRTVNET_TRT_INT8_ZERO_COND",
        _tensorrt_int8_agcm_only_enabled(mode_name)
        and _tensorrt_mode_name_is_full_int8(mode_name),
    )


def _tensorrt_int8_single_input_enabled(mode_name: str | None = None) -> bool:
    return (
        _tensorrt_int8_zero_condition_enabled(mode_name)
        and _env_bool(
            "HDRTVNET_TRT_INT8_SINGLE_INPUT",
            _tensorrt_int8_agcm_only_enabled(mode_name),
        )
    )


def _tensorrt_condition_free_arch_names_enabled(
    classifier: str | None,
    le_arch: str | None,
) -> bool:
    classifier = str(classifier or "color_condition").strip().lower().replace("-", "_")
    classifier_aliases = {
        "agcm_plain",
        "agcm_affine",
        "agcm_lite",
        "agcm_spatial",
        "lite_agcm",
        "adaptive_affine",
        "plain",
        "plain3",
        "plain_agcm",
        "plain_agcm3",
        "agcm_base",
        "agcm_base3",
        "base",
        "base3",
    }
    le_arch = str(le_arch or "sft").strip().lower().replace("-", "_")
    return (
        classifier in classifier_aliases
        or classifier.startswith("agcm_spatialh")
        or classifier.startswith("agcm_spatialmix")
    ) and (
        le_arch.startswith("plainflat")
        or le_arch.startswith("plainbottleneck")
        or le_arch.startswith("plaindirect")
        or le_arch.startswith("conddirect")
    )


def _tensorrt_condition_free_arch_enabled() -> bool:
    return _tensorrt_condition_free_arch_names_enabled(
        os.environ.get("HDRTVNET_CLASSIFIER", "color_condition"),
        os.environ.get("HDRTVNET_LE_ARCH", "sft"),
    )


def _tensorrt_condition_free_arch_enabled_for_checkpoint(model_path: str) -> bool:
    arch = _source_checkpoint_architecture(model_path)
    if arch:
        return _tensorrt_condition_free_arch_names_enabled(
            str(arch.get("classifier") or ""),
            str(arch.get("le_arch") or ""),
        )
    return _tensorrt_condition_free_arch_enabled()


def _tensorrt_condition_free_arch_enabled_for_model(model: nn.Module | None) -> bool:
    root = getattr(model, "_orig_mod", model)
    classifier = getattr(root, "_hdrtvnet_classifier_arch", None)
    le_arch = getattr(root, "_hdrtvnet_le_arch", None)
    if classifier or le_arch:
        return _tensorrt_condition_free_arch_names_enabled(classifier, le_arch)
    return _tensorrt_condition_free_arch_enabled()


def _tensorrt_calibrator_algorithm() -> str:
    text = str(os.environ.get("HDRTVNET_TRT_CALIBRATOR", "entropy")).strip().lower()
    if text in {"minmax", "min-max", "min_max"}:
        return "minmax"
    return "entropy"


def _tensorrt_native_int8_obey_constraints() -> bool:
    return _env_bool("HDRTVNET_TRT_NATIVE_INT8_OBEY", False)


def _tensorrt_native_int8_prefer_constraints() -> bool:
    return _env_bool("HDRTVNET_TRT_NATIVE_INT8_PREFER_CONSTRAINTS", False)


def _tensorrt_native_int8_checkpoint_policy_enabled() -> bool:
    return _env_bool("HDRTVNET_TRT_NATIVE_INT8_CHECKPOINT_POLICY", True)


def _tensorrt_native_int8_policy_obey() -> bool:
    return _env_bool("HDRTVNET_TRT_NATIVE_INT8_POLICY_OBEY", True)


def _tensorrt_native_int8_policy_int8_outputs() -> bool:
    return _env_bool("HDRTVNET_TRT_NATIVE_INT8_POLICY_INT8_OUTPUTS", False)


def _tensorrt_native_int8_policy_output_constraints() -> bool:
    return _env_bool("HDRTVNET_TRT_NATIVE_INT8_POLICY_OUTPUTS", False)


def _tensorrt_native_int8_calibrate_before_fusion() -> bool:
    return _env_bool("HDRTVNET_TRT_NATIVE_INT8_CALIBRATE_BEFORE_FUSION", True)


def _tensorrt_auto_qparam_calibration_cache() -> bool:
    return _env_bool("HDRTVNET_TRT_AUTO_QPARAM_CALIBRATION_CACHE", True)


def _tensorrt_calibration_cache_header(algorithm: str | None = None) -> str:
    algorithm = str(algorithm or _tensorrt_calibrator_algorithm()).strip().lower()
    try:
        import tensorrt as trt

        parts = str(getattr(trt, "__version__", "")).split(".")
        nums = [int(re.sub(r"\D.*$", "", p) or "0") for p in parts[:3]]
        while len(nums) < 3:
            nums.append(0)
        version_token = f"{nums[0]}{nums[1]:02d}{nums[2]:02d}"
    except Exception:
        version_token = "101601"
    if algorithm == "minmax":
        return f"TRT-{version_token}-MinMaxCalibration"
    return f"TRT-{version_token}-EntropyCalibration2"


def _read_tensorrt_calibration_cache_tensor_names(cache_path: str | None) -> set[str]:
    path = str(cache_path or "").strip()
    if not path:
        return set()
    try:
        text = pathlib.Path(path).read_text(encoding="ascii", errors="replace")
    except Exception:
        return set()
    names: set[str] = set()
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        name = line.split(":", 1)[0].strip()
        if name:
            names.add(name)
    return names


def _tensorrt_calibration_fingerprint(
    *,
    precision: str,
    qdq_fusion: str,
    calibration_dataset: str | None = None,
    calibration_video: str | None = None,
    calibration_frames: int | None = None,
    calibration_cache: str | None = None,
) -> dict[str, object] | None:
    if not str(precision or "").startswith("int8"):
        return None
    if _tensorrt_int8_modelopt_enabled():
        return None
    if _resolve_tensorrt_qdq_fusion(precision, qdq_fusion) != "native":
        return None

    frames = _resolve_tensorrt_calibration_frames(calibration_frames)
    algorithm = _tensorrt_calibrator_algorithm()
    cache = str(calibration_cache or "").strip() or None
    cache_entry = None
    if cache:
        cache_path = os.path.abspath(os.path.expanduser(cache))
        cache_entry = {
            "path": cache_path,
            "file": _hash_file_raw(cache_path),
        }
    if calibration_dataset:
        path = os.path.abspath(os.path.expanduser(str(calibration_dataset)))
        entry: dict[str, object] = {
            "kind": "dataset",
            "path": path,
            "frames": frames,
            "algorithm": algorithm,
            "cache": cache_entry,
        }
        try:
            paths = _tensorrt_calibration_image_paths(path)
            digest = hashlib.sha256()
            root = pathlib.Path(path)
            for item in paths:
                try:
                    rel = item.resolve().relative_to(root.resolve()).as_posix()
                except Exception:
                    rel = str(item.resolve())
                try:
                    stat = item.stat()
                    digest.update(rel.encode("utf-8", errors="replace"))
                    digest.update(b"\0")
                    digest.update(str(int(stat.st_size)).encode("ascii"))
                    digest.update(b"\0")
                    digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
                    digest.update(b"\0")
                except Exception:
                    digest.update(rel.encode("utf-8", errors="replace"))
                    digest.update(b"\0missing\0")
            entry.update({"items": len(paths), "fingerprint": digest.hexdigest()})
        except Exception as exc:
            entry.update({"items": None, "fingerprint_error": type(exc).__name__})
        return entry

    if calibration_video:
        path = os.path.abspath(os.path.expanduser(str(calibration_video)))
        return {
            "kind": "video",
            "path": path,
            "frames": frames,
            "algorithm": algorithm,
            "cache": cache_entry,
            "fingerprint": _hash_file_raw(path) if os.path.isfile(path) else None,
        }

    if cache_entry is not None:
        return {
            "kind": "cache",
            "cache": cache_entry,
        }

    return {
        "kind": "synthetic",
        "frames": frames,
        "algorithm": algorithm,
        "cache": None,
    }


def _prefer_tensorrt_native_int8_layers(
    network,
    trt,
    calibrated_tensors: set[str] | None = None,
) -> None:
    """Ask TensorRT to keep calibrated native PTQ compute layers in INT8."""
    try:
        output_names = {
            network.get_output(i).name
            for i in range(int(network.num_outputs))
            if network.get_output(i) is not None
        }
    except Exception:
        output_names = set()

    calibrated_tensors = set(calibrated_tensors or ())
    if not calibrated_tensors:
        print("TensorRT native INT8 constraints: skipped (no calibrated tensor names)")
        return

    preferred_types = {
        "CONVOLUTION",
        "DECONVOLUTION",
        "FULLY_CONNECTED",
        "MATRIX_MULTIPLY",
    }
    constrained_layers = 0
    constrained_outputs = 0
    skipped_layers = 0
    skipped_unscaled = 0
    try:
        layer_count = int(network.num_layers)
    except Exception:
        layer_count = 0
    for index in range(layer_count):
        layer = network.get_layer(index)
        layer_type = str(getattr(layer, "type", "")).split(".")[-1]
        if layer_type not in preferred_types:
            skipped_layers += 1
            continue
        try:
            input_names = [
                layer.get_input(input_index).name
                for input_index in range(int(layer.num_inputs))
                if layer.get_input(input_index) is not None
            ]
            layer_output_names = [
                layer.get_output(output_index).name
                for output_index in range(int(layer.num_outputs))
                if layer.get_output(output_index) is not None
            ]
        except Exception:
            skipped_layers += 1
            continue
        data_inputs = [
            name for name in input_names
            if name and not name.startswith("model.") and not name.startswith("_to_copy")
        ]
        if not layer_output_names or not all(name in calibrated_tensors for name in layer_output_names):
            skipped_unscaled += 1
            continue
        if data_inputs and not all(name in calibrated_tensors for name in data_inputs):
            skipped_unscaled += 1
            continue
        try:
            layer.precision = trt.int8
            constrained_layers += 1
        except Exception:
            skipped_layers += 1
            continue
        try:
            output_count = int(layer.num_outputs)
        except Exception:
            output_count = 0
        for output_index in range(output_count):
            try:
                tensor = layer.get_output(output_index)
                if tensor is None or tensor.name in output_names:
                    continue
                layer.set_output_type(output_index, trt.int8)
                constrained_outputs += 1
            except Exception:
                continue
    print(
        "TensorRT native INT8 constraints: "
        f"{constrained_layers} layer(s), {constrained_outputs} output tensor(s), "
        f"skipped {skipped_layers}, unscaled {skipped_unscaled}"
    )


def _onnx_weight_module_name(weight_name: str | None) -> str | None:
    if not weight_name:
        return None
    name = str(weight_name)
    if name.startswith("model."):
        name = name[len("model."):]
    for suffix in (".weight", ".bias"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return None


def _tensorrt_native_int8_onnx_policy(
    onnx_path: str,
    *,
    quant_type: str | None,
    w8a8_layers: set[str] | None,
) -> dict[str, str]:
    """Map ONNX compute node names to either INT8 or FP16 for native TRT PTQ."""
    try:
        import onnx
    except Exception as exc:
        print(f"TensorRT native INT8 policy skipped: ONNX unavailable: {exc}")
        return {}

    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        print(f"TensorRT native INT8 policy skipped: ONNX read failed: {exc}")
        return {}

    quant = str(quant_type or "").strip().lower()
    target_layers = set(w8a8_layers or ())
    policy: dict[str, str] = {}
    for node in model.graph.node:
        if node.op_type not in {"Conv", "Gemm", "MatMul"}:
            continue
        weight_name = node.input[1] if len(node.input) > 1 else None
        module_name = _onnx_weight_module_name(weight_name)
        if quant == "w8a8_mixed":
            precision = "int8" if module_name in target_layers else "fp16"
        elif quant == "w8a8_full":
            precision = "int8"
        else:
            continue
        if node.name:
            policy[str(node.name)] = precision

    if policy:
        counts = {
            "int8": sum(1 for value in policy.values() if value == "int8"),
            "fp16": sum(1 for value in policy.values() if value == "fp16"),
        }
        print(
            "TensorRT native INT8 checkpoint policy: "
            f"{counts['int8']} INT8 compute node(s), "
            f"{counts['fp16']} FP16 compute node(s)"
        )
    return policy


def _tensorrt_onnx_compute_nodes_for_module_patterns(
    onnx_path: str,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...] = (),
) -> list[str]:
    try:
        import onnx
    except Exception as exc:
        print(f"TensorRT ONNX node selection skipped: ONNX unavailable: {exc}")
        return []

    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:
        print(f"TensorRT ONNX node selection skipped: ONNX read failed: {exc}")
        return []

    selected: list[str] = []
    total = 0
    skipped_unnamed = 0
    for node in model.graph.node:
        if node.op_type not in {"Conv", "Gemm", "MatMul"}:
            continue
        total += 1
        weight_name = node.input[1] if len(node.input) > 1 else None
        module_name = _onnx_weight_module_name(weight_name)
        if not module_name or not node.name:
            skipped_unnamed += 1
            continue
        keep = True
        if include:
            keep = _tensorrt_modelopt_torch_pattern_match(
                module_name,
                module_name,
                include,
            )
        if keep and exclude:
            keep = not _tensorrt_modelopt_torch_pattern_match(
                module_name,
                module_name,
                exclude,
            )
        if keep:
            selected.append(str(node.name))

    print(
        "TensorRT ONNX node selection: "
        f"{len(selected)}/{total} compute node(s), "
        f"include={list(include) or '*'}, exclude={list(exclude) or 'none'}, "
        f"skipped={skipped_unnamed}"
    )
    if selected:
        preview = ", ".join(selected[:12])
        more = "" if len(selected) <= 12 else f", +{len(selected) - 12} more"
        print(f"TensorRT ONNX selected nodes: {preview}{more}")
    return selected


def _tensorrt_layer_onnx_names(layer) -> set[str]:
    text_parts = [
        str(getattr(layer, "name", "") or ""),
        str(getattr(layer, "metadata", "") or ""),
    ]
    names: set[str] = set()
    for text in text_parts:
        names.update(re.findall(r"\bnode_(?:conv2d|linear|gemm|matmul)[A-Za-z0-9_]*\b", text))
    return names


def _apply_tensorrt_native_int8_checkpoint_policy(
    network,
    trt,
    policy: dict[str, str],
    *,
    int8_outputs: bool,
    constrain_outputs: bool,
) -> tuple[int, int, int]:
    if not policy:
        return 0, 0, 0

    output_names: set[str] = set()
    try:
        output_names = {
            network.get_output(i).name
            for i in range(int(network.num_outputs))
            if network.get_output(i) is not None
        }
    except Exception:
        output_names = set()

    int8_layers = 0
    fp16_layers = 0
    output_constraints = 0
    preferred_types = {
        "CONVOLUTION",
        "DECONVOLUTION",
        "FULLY_CONNECTED",
        "MATRIX_MULTIPLY",
    }
    try:
        layer_count = int(network.num_layers)
    except Exception:
        layer_count = 0

    for index in range(layer_count):
        layer = network.get_layer(index)
        layer_type = str(getattr(layer, "type", "")).split(".")[-1]
        onnx_names = _tensorrt_layer_onnx_names(layer)
        matches = [policy[name] for name in onnx_names if name in policy]
        if not matches:
            continue
        precision = "int8" if "int8" in matches else "fp16"
        try:
            if precision == "int8":
                layer.precision = trt.int8
                int8_layers += 1
            else:
                layer.precision = trt.float16
                fp16_layers += 1
        except Exception:
            continue

        if not constrain_outputs or layer_type not in preferred_types:
            continue
        try:
            output_count = int(layer.num_outputs)
        except Exception:
            output_count = 0
        for output_index in range(output_count):
            try:
                tensor = layer.get_output(output_index)
                if tensor is None or tensor.name in output_names:
                    continue
                layer.set_output_type(
                    output_index,
                    trt.int8 if (precision == "int8" and int8_outputs) else trt.float16,
                )
                output_constraints += 1
            except Exception:
                continue

    print(
        "TensorRT native INT8 checkpoint policy applied: "
        f"{int8_layers} INT8 layer(s), {fp16_layers} FP16 layer(s), "
        f"{output_constraints} output constraint(s)"
    )
    return int8_layers, fp16_layers, output_constraints


def tensorrt_engine_path(
    model_path: str,
    width: int,
    height: int,
    mode: str,
) -> str:
    model_path = tensorrt_source_checkpoint_path(model_path)
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
    model_path = tensorrt_source_checkpoint_path(model_path)
    return os.path.splitext(tensorrt_engine_path(model_path, width, height, mode))[0] + ".onnx"


def cleanup_tensorrt_onnx_after_engine(onnx_path: str, engine_path: str) -> bool:
    if not onnx_path or not os.path.isfile(engine_path):
        return False
    if _env_bool("HDRTVNET_TRT_KEEP_ONNX", False):
        print(f"TensorRT ONNX kept for inspection: {onnx_path}")
        return False
    try:
        removed = False
        if os.path.isfile(onnx_path):
            os.remove(onnx_path)
            removed = True
        data_path = f"{onnx_path}.data"
        if os.path.isfile(data_path):
            os.remove(data_path)
            removed = True
        if removed:
            print(f"TensorRT ONNX removed after engine ready: {onnx_path}")
        return removed
    except Exception as exc:
        print(f"TensorRT ONNX cleanup skipped: {exc}")
        return False


def _resolve_tensorrt_predequantize(precision: str, predequantize=False) -> bool:
    if not str(precision or "").startswith("int8"):
        return False
    if (
        str(precision or "").startswith("int8-full")
        and _tensorrt_full_int8_fp16_islands_enabled()
        and _tensorrt_full_int8_safe_fp16_builder_enabled()
    ):
        return True
    if isinstance(predequantize, bool):
        return bool(predequantize)
    text = str(predequantize or "off").strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    if str(precision or "").startswith("int8-mixed") and _env_bool(
        "HDRTVNET_TRT_MIXED_FP16_BUILDER",
        False,
    ):
        return True
    # Auto mirrors the PyTorch path: keep explicit INT8 only on NVIDIA GPUs
    # with native INT8 tensor cores; otherwise export a native FP16 engine.
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            has_int8_tc = props.major > 7 or (props.major == 7 and props.minor >= 5)
            return not has_int8_tc
        except Exception:
            return False
    return False


def tensorrt_mode_name(
    precision: str,
    mode: str,
    predequantize=False,
    qdq_fusion: str = "auto",
) -> str:
    mode_name = str(mode or precision or "mode")
    if str(precision).startswith("int8"):
        lower = mode_name.lower()
        if _resolve_tensorrt_predequantize(precision, predequantize):
            if str(precision).startswith("int8-full"):
                has_island_suffix = (
                    "fpislandoff" in lower or "fpislandson" in lower
                )
                if (
                    _tensorrt_full_int8_fp16_islands_enabled()
                    and not has_island_suffix
                ):
                    mode_name = f"{mode_name}_fpislandsonv1"
                    lower = mode_name.lower()
            if "predeq" not in lower:
                mode_name = f"{mode_name}_predeqv1"
                lower = mode_name.lower()
            if (
                _tensorrt_int8_agcm_only_enabled(mode_name)
                and "agcmonly" not in lower
            ):
                mode_name = f"{mode_name}_agcmonlyv1"
            return mode_name
        if _tensorrt_int8_modelopt_enabled():
            use_qat_checkpoint = (
                _tensorrt_int8_qat_checkpoint_composition_enabled(mode_name)
            )
            use_torch_modelopt = (
                _tensorrt_int8_modelopt_torch_enabled()
                and not use_qat_checkpoint
            )
            suffix = (
                "modelopttorchint8"
                if use_torch_modelopt
                else "modeloptint8"
            )
            if suffix not in lower:
                mode_name = f"{mode_name}_{suffix}v1"
                lower = mode_name.lower()
            if use_qat_checkpoint and "qatckpt" not in lower:
                mode_name = f"{mode_name}_qatckptv1"
                lower = mode_name.lower()
            if (
                use_torch_modelopt
                and _tensorrt_int8_modelopt_torch_quant_scheme() == "asymmetric"
                and "asymact" not in lower
            ):
                mode_name = f"{mode_name}_asymactv1"
            return mode_name
        fusion = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
        if fusion == "native":
            if "nativeint8" not in lower and "implicitint8" not in lower:
                mode_name = f"{mode_name}_nativeint8v1"
                lower = mode_name.lower()
            if str(precision).startswith("int8-full"):
                has_island_suffix = (
                    "fpislandoff" in lower or "fpislandson" in lower
                )
                if not has_island_suffix:
                    suffix = (
                        "_fpislandsonv1"
                        if _tensorrt_full_int8_fp16_islands_enabled()
                        else "_fpislandoffv1"
                    )
                    mode_name = f"{mode_name}{suffix}"
                    lower = mode_name.lower()
            if (
                _tensorrt_int8_agcm_only_enabled(mode_name)
                and "agcmonly" not in lower
            ):
                mode_name = f"{mode_name}_agcmonlyv1"
                lower = mode_name.lower()
            if (
                _tensorrt_int8_single_input_enabled(mode_name)
                and "singleinput" not in lower
            ):
                mode_name = f"{mode_name}_singleinputv1"
            return mode_name
        if "qdq" in lower:
            return mode_name
        qdq_version = "qdqv7"
        mode_name = f"{mode_name}_{qdq_version}"
        if fusion == "add" and "addqdq" not in lower:
            mode_name = f"{mode_name}_addqdqv1"
        elif fusion == "add-mul" and "addmulqdq" not in lower:
            mode_name = f"{mode_name}_addmulqdqv1"
        fp_suffix = _tensorrt_fp_export_policy_suffix()
        if fp_suffix and fp_suffix not in mode_name.lower():
            mode_name = f"{mode_name}{fp_suffix}"
    return mode_name


def tensorrt_prebuilt_calibration_cache_path(
    model_path: str,
    width: int,
    height: int,
    precision: str,
    mode_name: str,
    *,
    use_hg: bool,
    predequantize=False,
    qdq_fusion: str = "native",
    calibration_dir: str | None = None,
    require_exists: bool = True,
) -> str | None:
    """Return the shipped TensorRT native-INT8 calibration cache path.

    The filename mirrors the engine stem so each checkpoint/resolution/mode
    combination can have its own cache without a separate manifest.
    """
    if not str(precision or "").startswith("int8"):
        return None
    if _tensorrt_int8_modelopt_enabled():
        return None
    predeq = _resolve_tensorrt_predequantize(precision, predequantize)
    if predeq:
        return None
    resolved_qdq = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
    if resolved_qdq != "native":
        return None
    agcm_only = _tensorrt_int8_agcm_only_enabled(mode_name)

    mode = tensorrt_mode_name(
        precision,
        mode_name,
        predequantize=predeq,
        qdq_fusion=resolved_qdq,
    )
    model_name = _sanitize_engine_token(pathlib.Path(str(model_path)).stem)
    resolution = f"{int(width)}x{int(height)}"
    mode_token = _sanitize_engine_token(mode)
    calib_name = f"{model_name}_{resolution}_{mode_token}.calib"
    root = (
        os.path.abspath(os.path.expanduser(str(calibration_dir)))
        if calibration_dir
        else _tensorrt_calibration_dir(create=not require_exists)
    )
    path = os.path.join(root, calib_name)
    if require_exists and not os.path.isfile(path):
        if _tensorrt_auto_qparam_calibration_cache() and not agcm_only:
            try:
                build_tensorrt_w8a8_qparam_calibration_cache(
                    model_path,
                    width,
                    height,
                    precision,
                    mode_name,
                    use_hg=use_hg,
                    cache_path=path,
                    predequantize=False,
                    qdq_fusion="native",
                    keep_onnx=False,
                    force_onnx=False,
                )
            except Exception as exc:
                print(
                    "TensorRT native qparam calibration cache generation skipped: "
                    f"{exc}"
                )
        if os.path.isfile(path):
            return path
        return None
    return path


def _resolve_tensorrt_calibration_sources(
    *,
    model_path: str,
    width: int,
    height: int,
    precision: str,
    mode_name: str,
    use_hg: bool,
    predequantize=False,
    qdq_fusion: str = "native",
    calibration_dataset: str | None = None,
    calibration_video: str | None = None,
    calibration_cache: str | None = None,
) -> tuple[str | None, str | None, str | None, bool]:
    dataset = (
        calibration_dataset
        or os.environ.get("HDRTVNET_TRT_CALIBRATION_DATASET")
        or os.environ.get("HDRTVNET_TRT_CALIBRATION_DIR")
    )
    video = calibration_video or os.environ.get("HDRTVNET_TRT_CALIBRATION_VIDEO")
    cache = calibration_cache or os.environ.get("HDRTVNET_TRT_CALIBRATION_CACHE")
    used_prebuilt = False
    if str(precision or "").startswith("int8") and _tensorrt_int8_modelopt_enabled():
        return None, None, None, False
    if not dataset and not video and not cache:
        cache = tensorrt_prebuilt_calibration_cache_path(
            model_path,
            int(width),
            int(height),
            precision,
            mode_name,
            use_hg=use_hg,
            predequantize=predequantize,
            qdq_fusion=qdq_fusion,
            require_exists=True,
        )
        used_prebuilt = bool(cache)
    return dataset, video, cache, used_prebuilt


def _tensorrt_w8_module_weight(module: nn.Module) -> np.ndarray | None:
    if isinstance(module, W8A8Conv2d):
        scale = module.w_scale.detach().float().cpu().view(-1, 1, 1, 1)
        return (module.weight_int8.detach().float().cpu() * scale).numpy()
    if isinstance(module, W8Conv2d):
        scale = module.scale.detach().float().cpu().view(-1, 1, 1, 1)
        return (module.weight_int8.detach().float().cpu() * scale).numpy()
    if isinstance(module, nn.Conv2d):
        return module.weight.detach().float().cpu().numpy()
    return None


def _tensorrt_w8a8_activation_range(module: nn.Module) -> float | None:
    if not isinstance(module, (W8A8Conv2d, W8A8Linear)):
        return None
    scale = float(module.x_scale.detach().float().cpu().reshape(()))
    if bool(getattr(module, "is_asymmetric", False)):
        zero = getattr(module, "x_zero", torch.tensor(0.0))
        low = float(zero.detach().float().cpu().reshape(()))
        high = low + scale * 255.0
        return max(abs(low), abs(high), 1e-5)
    return max(abs(scale * 127.0), 1e-5)


def _onnx_initializer_array(model, tensor_name: str):
    try:
        from onnx import numpy_helper
    except Exception:
        return None

    initializers = {init.name: init for init in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output}

    def _array(name: str):
        init = initializers.get(name)
        if init is not None:
            return numpy_helper.to_array(init)
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type == "Cast" and node.input:
            return _array(node.input[0])
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    return numpy_helper.to_array(attr.t)
        return None

    return _array(tensor_name)


def _write_tensorrt_calibration_cache(
    cache_path: str,
    ranges: dict[str, float],
    *,
    algorithm: str | None = None,
) -> None:
    if not ranges:
        raise RuntimeError("No TensorRT calibration ranges were generated.")
    path = pathlib.Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_tensorrt_calibration_cache_header(algorithm)]
    for name in sorted(ranges):
        value = max(float(ranges[name]), 1e-8)
        lines.append(f"{name}: {struct.pack('>f', value).hex()}")
    path.write_text("\n".join(lines), encoding="ascii")


def _tensorrt_initializer_ranges(onnx_model) -> dict[str, float]:
    try:
        from onnx import numpy_helper
    except Exception:
        return {}
    ranges: dict[str, float] = {}
    for initializer in getattr(onnx_model.graph, "initializer", []):
        try:
            array = numpy_helper.to_array(initializer).astype(np.float32, copy=False)
            value = float(np.max(np.abs(array))) if array.size else 1.0
        except Exception:
            value = 1.0
        value = max(value, 1e-5)
        ranges[initializer.name] = value
        ranges[f"{initializer.name}_output"] = value
    return ranges


def _tensorrt_all_graph_tensor_ranges(
    onnx_model,
    *,
    default_range: float,
) -> dict[str, float]:
    value = max(float(default_range), 1e-5)
    ranges: dict[str, float] = {}
    for collection in (
        getattr(onnx_model.graph, "input", []),
        getattr(onnx_model.graph, "output", []),
        getattr(onnx_model.graph, "value_info", []),
    ):
        for item in collection:
            name = str(getattr(item, "name", "") or "")
            if name:
                ranges[name] = value
    for node in getattr(onnx_model.graph, "node", []):
        for name in list(node.input) + list(node.output):
            if name:
                ranges.setdefault(str(name), value)
    ranges.update(_tensorrt_initializer_ranges(onnx_model))
    ranges["input"] = max(float(ranges.get("input", value)), 1.0)
    ranges["cond"] = max(float(ranges.get("cond", value)), 1.0)
    ranges["output"] = max(float(ranges.get("output", value)), 1.0)
    return ranges


def _tensorrt_collect_runtime_module_ranges(
    processor: "HDRTVNetTorch",
    *,
    width: int,
    height: int,
    calibration_dataset: str | None,
    calibration_video: str | None,
    calibration_frames: int | None,
) -> tuple[dict[str, float], int]:
    module_ranges: dict[str, float] = {}
    hook_types = (
        W8Conv2d,
        W8A8Conv2d,
        W8Linear,
        W8A8Linear,
        nn.Conv2d,
        nn.Linear,
        nn.ReLU,
        nn.LeakyReLU,
        nn.PixelShuffle,
        nn.InstanceNorm2d,
        nn.AvgPool2d,
        nn.AdaptiveAvgPool2d,
    )
    handles = []

    def _record(name: str, output) -> None:
        if isinstance(output, (tuple, list)):
            if not output:
                return
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return
        try:
            value = float(output.detach().float().abs().amax().cpu().item())
        except Exception:
            return
        if math.isfinite(value):
            module_ranges[name] = max(float(module_ranges.get(name, 0.0)), value, 1e-5)

    for name, module in processor.model.named_modules():
        if isinstance(module, hook_types):
            handles.append(module.register_forward_hook(lambda _m, _i, o, n=name: _record(n, o)))

    source: _TensorRTCalibrationSource | None = None
    try:
        frames = _resolve_tensorrt_calibration_frames(calibration_frames)
        if calibration_dataset:
            source = _make_tensorrt_dataset_calibration_source(
                dataset_path=calibration_dataset,
                width=width,
                height=height,
                dtype=processor._dtype,
                device=processor.device,
                frame_count=frames,
            )
        elif calibration_video:
            source = _make_tensorrt_video_calibration_source(
                video_path=calibration_video,
                width=width,
                height=height,
                dtype=processor._dtype,
                device=processor.device,
                frame_count=frames,
            )
        else:
            source = _make_synthetic_tensorrt_calibration_source(
                width=width,
                height=height,
                dtype=processor._dtype,
                device=processor.device,
                frame_count=max(8, frames),
            )

        seen = 0
        with torch.inference_mode():
            while True:
                batch = source.next_batch()
                if batch is None:
                    break
                tensor = batch.get("input")
                cond = batch.get("cond")
                if tensor is None or cond is None:
                    continue
                if getattr(processor, "_is_flat_model", False):
                    processor.model(tensor, cond)
                else:
                    processor.model((tensor, cond))
                seen += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize(processor.device)
        return module_ranges, seen
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


def _tensorrt_weight_module_entries(processor: "HDRTVNetTorch") -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for name, module in processor.model.named_modules():
        weight = _tensorrt_w8_module_weight(module)
        if weight is None:
            continue
        entries.append(
            {
                "name": name,
                "module": module,
                "weight": weight,
                "range": _tensorrt_w8a8_activation_range(module),
            }
        )
    return entries


def _tensorrt_map_conv_nodes_to_modules(
    onnx_model,
    modules: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    conv_nodes = [node for node in onnx_model.graph.node if node.op_type == "Conv"]
    used_modules: set[int] = set()
    mappings: list[dict[str, object]] = []
    failures: list[str] = []

    for node in conv_nodes:
        if len(node.input) < 2:
            failures.append(f"{node.name or node.output[0]}: missing weight input")
            continue
        onnx_weight = _onnx_initializer_array(onnx_model, node.input[1])
        if onnx_weight is None:
            failures.append(f"{node.name or node.output[0]}: unresolved weight tensor")
            continue
        weight = onnx_weight.astype(np.float32, copy=False)
        best: tuple[float, int, dict[str, object]] | None = None
        for index, entry in enumerate(modules):
            if index in used_modules:
                continue
            candidate = entry["weight"]
            if candidate.shape != weight.shape:
                continue
            diff = float(np.max(np.abs(weight - candidate)))
            if best is None or diff < best[0]:
                best = (diff, index, entry)
        if best is None or best[0] > 1e-2:
            label = node.name or (node.output[0] if node.output else "<conv>")
            failures.append(f"{label}: no checkpoint weight match")
            continue
        diff, index, entry = best
        used_modules.add(index)
        mappings.append(
            {
                "node": node,
                "node_name": node.name or (node.output[0] if node.output else "<conv>"),
                "input": node.input[0] if node.input else "",
                "output": node.output[0] if node.output else "",
                "weight": node.input[1] if len(node.input) > 1 else "",
                "bias": node.input[2] if len(node.input) > 2 else "",
                "module": str(entry["name"]),
                "module_entry": entry,
                "weight_max_abs_diff": diff,
            }
        )
    return mappings, failures


def _tensorrt_propagate_calibration_ranges(
    onnx_model,
    ranges: dict[str, float],
    *,
    passes: int = 4,
    cap: float = 1024.0,
) -> None:
    alias_ops = {
        "Relu",
        "LeakyRelu",
        "Identity",
        "Cast",
        "Transpose",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "DepthToSpace",
        "AveragePool",
        "GlobalAveragePool",
        "ReduceMean",
        "Slice",
        "Gather",
    }
    for _ in range(max(1, int(passes))):
        changed = False
        for node in onnx_model.graph.node:
            outputs = [name for name in node.output if name]
            if not outputs:
                continue
            input_values = [
                float(ranges[name])
                for name in node.input
                if name and name in ranges and math.isfinite(float(ranges[name]))
            ]
            if not input_values:
                continue
            op = str(node.op_type)
            if op in alias_ops:
                value = max(input_values)
            elif op in {"Sigmoid", "Tanh"}:
                value = 1.0
            elif op in {"Add", "Sub", "Sum", "Concat"}:
                value = sum(input_values)
            elif op == "Mul":
                if len(input_values) >= 2:
                    value = input_values[0]
                    for item in input_values[1:]:
                        value *= item
                else:
                    value = input_values[0]
            elif op == "Div":
                value = input_values[0]
            elif op == "Clip":
                value = input_values[0]
            else:
                continue
            value = min(max(float(value), 1e-5), float(cap))
            for output in outputs:
                if value > float(ranges.get(output, 0.0)) + 1e-8:
                    ranges[output] = value
                    changed = True
        if not changed:
            break


def _tensorrt_agcm_protected_tensors(onnx_model) -> set[str]:
    producers = {
        output: node
        for node in onnx_model.graph.node
        for output in node.output
        if output
    }
    protected: set[str] = set()

    def _protect_ancestors(name: str) -> None:
        if not name or name in protected or name in {"input", "cond"}:
            return
        protected.add(name)
        node = producers.get(name)
        if node is None:
            return
        for item in list(node.input) + list(node.output):
            if item and item not in {"input", "cond"}:
                _protect_ancestors(item)

    # In the native no-HG export, add_5 is the AGCM output handed to LE.  Leaving
    # that ancestor path uncalibrated keeps TensorRT in FP16 for the conditioning
    # branch while allowing the LE trunk to use contiguous native INT8 tactics.
    for boundary in ("add_5",):
        if boundary in producers:
            _protect_ancestors(boundary)

    agcm_patterns = (
        "model.AGCM",
        "node_linear",
        "linear",
        "view",
        "ONNXTRT_unsqueezeTensor",
        "instance_norm",
    )
    for name in list(producers):
        if name not in {"input", "cond"} and any(name.startswith(p) for p in agcm_patterns):
            _protect_ancestors(name)
    for initializer in getattr(onnx_model.graph, "initializer", []):
        name = str(getattr(initializer, "name", "") or "")
        if name.startswith("model.AGCM"):
            protected.add(name)
            protected.add(f"{name}_output")
    return protected - {"input", "cond"}


def build_tensorrt_native_int8_speed_calibration_cache(
    model_path: str,
    width: int,
    height: int,
    precision: str,
    mode_name: str,
    *,
    use_hg: bool,
    cache_path: str | None = None,
    predequantize=False,
    qdq_fusion: str = "native",
    device: str = "auto",
    hg_weights: str | None = None,
    keep_onnx: bool = False,
    force_onnx: bool = True,
    calibration_dataset: str | None = None,
    calibration_video: str | None = None,
    calibration_frames: int | None = 64,
    default_range: float = 1.0,
    protect_agcm: bool = True,
) -> dict[str, object]:
    """Build the fast native TensorRT INT8 cache from PyTorch runtime ranges.

    The cache is still TensorRT implicit/native INT8, not Q/DQ.  Runtime ranges
    are measured from the quantized PyTorch checkpoint, ONNX coverage is filled
    broadly enough for TensorRT to form INT8 tactic regions, and the AGCM
    conditioning branch is intentionally left uncalibrated for visual parity.
    """
    if not str(precision or "").startswith("int8"):
        raise ValueError("Native INT8 speed caches require an INT8 precision.")
    predeq = _resolve_tensorrt_predequantize(precision, predequantize)
    if predeq:
        raise ValueError("Native INT8 speed caches require predequantize=off.")
    resolved_qdq = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
    if resolved_qdq != "native":
        raise ValueError("Native INT8 speed caches are native TensorRT only.")

    model_path = tensorrt_source_checkpoint_path(model_path)
    cache_path = cache_path or tensorrt_prebuilt_calibration_cache_path(
        model_path,
        width,
        height,
        precision,
        mode_name,
        use_hg=use_hg,
        predequantize=False,
        qdq_fusion="native",
        require_exists=False,
    )
    if not cache_path:
        raise RuntimeError("Could not resolve TensorRT calibration cache path.")

    engine_mode = tensorrt_mode_name(
        precision,
        mode_name,
        predequantize=False,
        qdq_fusion="native",
    )
    onnx_path = tensorrt_onnx_path(model_path, width, height, engine_mode)
    if force_onnx:
        for candidate in (onnx_path, f"{onnx_path}.data"):
            try:
                if os.path.isfile(candidate):
                    os.remove(candidate)
            except OSError:
                pass

    processor = HDRTVNetTorch(
        model_path,
        device=device,
        precision=precision,
        compile_model=False,
        predequantize=False,
        hg_weights=hg_weights,
        use_hg=use_hg,
        warmup_passes=0,
    )
    processor._configure_assume_aligned_shapes(width, height)
    module_entries = _tensorrt_weight_module_entries(processor)
    module_ranges, frame_count = _tensorrt_collect_runtime_module_ranges(
        processor,
        width=width,
        height=height,
        calibration_dataset=calibration_dataset,
        calibration_video=calibration_video,
        calibration_frames=calibration_frames,
    )

    _export_tensorrt_onnx_from_model(
        model=processor.model,
        onnx_path=onnx_path,
        width=width,
        height=height,
        dtype=processor._dtype,
        device=processor.device,
        precision=precision,
        flat_model=getattr(processor, "_is_flat_model", False),
        qdq_fusion="native",
    )

    try:
        import onnx
    except Exception as exc:
        raise RuntimeError("onnx is required to build native INT8 speed caches.") from exc
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    ranges = _tensorrt_all_graph_tensor_ranges(
        onnx_model,
        default_range=max(float(default_range), 1e-5),
    )
    mappings, failures = _tensorrt_map_conv_nodes_to_modules(onnx_model, module_entries)
    if failures:
        sample = "\n".join(f"  - {item}" for item in failures[:8])
        raise RuntimeError(
            "Could not map every TensorRT Conv node to checkpoint weights:\n"
            f"{sample}"
        )

    mapped = 0
    for item in mappings:
        module_name = str(item["module"])
        output_range = float(module_ranges.get(module_name, 0.0))
        entry = item["module_entry"]
        qparam_range = entry.get("range") if isinstance(entry, dict) else None
        if output_range <= 0.0 and qparam_range is not None:
            output_range = float(qparam_range)
        if output_range <= 0.0:
            output_range = max(float(default_range), 1e-5)
        input_name = str(item.get("input") or "")
        output_name = str(item.get("output") or "")
        weight_name = str(item.get("weight") or "")
        bias_name = str(item.get("bias") or "")
        if input_name:
            ranges[input_name] = max(float(ranges.get(input_name, 0.0)), 1.0)
        if output_name:
            ranges[output_name] = max(float(ranges.get(output_name, 0.0)), output_range)
        if weight_name:
            weight_range = _onnx_initializer_array(onnx_model, weight_name)
            if weight_range is not None:
                weight_value = float(np.max(np.abs(weight_range.astype(np.float32, copy=False))))
                ranges[weight_name] = max(weight_value, 1e-5)
                ranges[f"{weight_name}_output"] = max(weight_value, 1e-5)
        if bias_name:
            bias_array = _onnx_initializer_array(onnx_model, bias_name)
            if bias_array is not None:
                bias_value = float(np.max(np.abs(bias_array.astype(np.float32, copy=False))))
                ranges[bias_name] = max(bias_value, 1e-5)
                ranges[f"{bias_name}_output"] = max(bias_value, 1e-5)
        mapped += 1

    _tensorrt_propagate_calibration_ranges(onnx_model, ranges)
    protected_count = 0
    if protect_agcm:
        protected = _tensorrt_agcm_protected_tensors(onnx_model)
        protected_count = len(protected)
        for name in protected:
            ranges.pop(name, None)

    _write_tensorrt_calibration_cache(cache_path, ranges)
    if not keep_onnx:
        cleanup_tensorrt_onnx_after_engine(onnx_path, cache_path)

    processor.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "TensorRT native INT8 speed calibration cache: "
        f"{cache_path} ({frame_count} frame(s), {mapped} Conv, "
        f"{len(ranges)} tensor range(s), protected={protected_count})"
    )
    return {
        "cache_path": os.path.abspath(cache_path),
        "range_count": len(ranges),
        "conv_count": mapped,
        "frames": int(frame_count),
        "protected_count": int(protected_count),
        "onnx_path": os.path.abspath(onnx_path),
        "mappings": [
            {
                "node": str(item.get("node_name") or ""),
                "input": str(item.get("input") or ""),
                "output": str(item.get("output") or ""),
                "module": str(item.get("module") or ""),
                "weight_max_abs_diff": float(item.get("weight_max_abs_diff") or 0.0),
            }
            for item in mappings
        ],
    }


def build_tensorrt_w8a8_qparam_calibration_cache(
    model_path: str,
    width: int,
    height: int,
    precision: str,
    mode_name: str,
    *,
    use_hg: bool,
    cache_path: str | None = None,
    predequantize=False,
    qdq_fusion: str = "native",
    device: str = "auto",
    hg_weights: str | None = None,
    keep_onnx: bool = False,
    force_onnx: bool = True,
    output_range_floor: float = 1.0,
) -> dict[str, object]:
    """Build a native TensorRT INT8 cache from checkpoint W8A8 qparams.

    This does not add Q/DQ nodes.  It exports the native TensorRT ONNX graph,
    matches each ONNX Conv weight tensor back to its checkpoint module, and
    writes ranges only for real W8A8 activation tensors.  W8A16/FP layers are
    left without activation ranges so TensorRT can keep them as native FP
    tactics instead of manufacturing unrelated INT8 islands.
    """
    if not str(precision or "").startswith("int8"):
        raise ValueError("W8A8 qparam calibration caches require an INT8 precision.")
    predeq = _resolve_tensorrt_predequantize(precision, predequantize)
    if predeq:
        raise ValueError("W8A8 qparam calibration caches require predequantize=off.")
    resolved_qdq = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
    if resolved_qdq != "native":
        raise ValueError("W8A8 qparam calibration caches are native TensorRT only.")

    model_path = tensorrt_source_checkpoint_path(model_path)
    cache_path = cache_path or tensorrt_prebuilt_calibration_cache_path(
        model_path,
        width,
        height,
        precision,
        mode_name,
        use_hg=use_hg,
        predequantize=False,
        qdq_fusion="native",
        require_exists=False,
    )
    if not cache_path:
        raise RuntimeError("Could not resolve TensorRT calibration cache path.")

    engine_mode = tensorrt_mode_name(
        precision,
        mode_name,
        predequantize=False,
        qdq_fusion="native",
    )
    onnx_path = tensorrt_onnx_path(model_path, width, height, engine_mode)
    if force_onnx:
        _remove_candidates = [onnx_path, f"{onnx_path}.data"]
        for candidate in _remove_candidates:
            try:
                if os.path.isfile(candidate):
                    os.remove(candidate)
            except OSError:
                pass

    processor = HDRTVNetTorch(
        model_path,
        device=device,
        precision=precision,
        compile_model=False,
        predequantize=False,
        hg_weights=hg_weights,
        use_hg=use_hg,
        warmup_passes=0,
    )
    processor._configure_assume_aligned_shapes(width, height)
    modules = []
    for name, module in processor.model.named_modules():
        weight = _tensorrt_w8_module_weight(module)
        if weight is not None:
            modules.append(
                {
                    "name": name,
                    "module": module,
                    "weight": weight,
                    "range": _tensorrt_w8a8_activation_range(module),
                }
            )

    _export_tensorrt_onnx_from_model(
        model=processor.model,
        onnx_path=onnx_path,
        width=width,
        height=height,
        dtype=processor._dtype,
        device=processor.device,
        precision=precision,
        flat_model=getattr(processor, "_is_flat_model", False),
        qdq_fusion="native",
    )

    try:
        import onnx
    except Exception as exc:
        raise RuntimeError("onnx is required to build W8A8 qparam caches.") from exc
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    conv_nodes = [node for node in onnx_model.graph.node if node.op_type == "Conv"]
    used_modules: set[int] = set()
    ranges: dict[str, float] = {"input": 1.0, "cond": 1.0}
    mappings: list[dict[str, object]] = []
    failures: list[str] = []

    for node in conv_nodes:
        if len(node.input) < 2:
            failures.append(f"{node.name or node.output[0]}: missing weight input")
            continue
        onnx_weight = _onnx_initializer_array(onnx_model, node.input[1])
        if onnx_weight is None:
            failures.append(f"{node.name or node.output[0]}: unresolved weight tensor")
            continue
        weight = onnx_weight.astype(np.float32, copy=False)
        best: tuple[float, int, dict[str, object]] | None = None
        for index, entry in enumerate(modules):
            if index in used_modules:
                continue
            candidate = entry["weight"]
            if candidate.shape != weight.shape:
                continue
            diff = float(np.max(np.abs(weight - candidate)))
            if best is None or diff < best[0]:
                best = (diff, index, entry)
        if best is None or best[0] > 1e-2:
            label = node.name or (node.output[0] if node.output else "<conv>")
            failures.append(f"{label}: no checkpoint weight match")
            continue
        diff, index, entry = best
        used_modules.add(index)
        activation_range = entry.get("range")
        if activation_range is None:
            continue
        value = float(activation_range)
        input_name = node.input[0]
        output_name = node.output[0]
        ranges[input_name] = max(float(ranges.get(input_name, 0.0)), value)
        ranges[output_name] = max(
            float(ranges.get(output_name, 0.0)),
            max(float(output_range_floor), value),
        )
        mappings.append(
            {
                "node": node.name or output_name,
                "input": input_name,
                "output": output_name,
                "module": str(entry["name"]),
                "range": value,
                "weight_max_abs_diff": diff,
            }
        )

    if failures:
        sample = "\n".join(f"  - {item}" for item in failures[:8])
        raise RuntimeError(
            "Could not map every TensorRT Conv node to checkpoint weights:\n"
            f"{sample}"
        )
    if not mappings:
        raise RuntimeError("No W8A8 Conv modules were found for TensorRT calibration.")

    _write_tensorrt_calibration_cache(cache_path, ranges)
    if not keep_onnx:
        cleanup_tensorrt_onnx_after_engine(onnx_path, cache_path)

    processor.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "TensorRT native W8A8 qparam calibration cache: "
        f"{cache_path} ({len(mappings)} W8A8 Conv, {len(ranges)} tensor range(s))"
    )
    return {
        "cache_path": os.path.abspath(cache_path),
        "range_count": len(ranges),
        "w8a8_conv_count": len(mappings),
        "onnx_path": os.path.abspath(onnx_path),
        "mappings": mappings,
    }


class _ONNXExportWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        flat_model: bool = False,
        zero_cond: torch.Tensor | None = None,
        agcm_only: bool = False,
        allow_missing_cond: bool = False,
    ):
        super().__init__()
        self.model = model
        self.flat_model = bool(flat_model)
        self.agcm_only = bool(agcm_only)
        self.allow_missing_cond = bool(allow_missing_cond)
        if zero_cond is not None:
            self.register_buffer("_zero_cond", zero_cond, persistent=False)
        else:
            self._zero_cond = None

    def forward(self, tensor: torch.Tensor, cond: torch.Tensor | None = None):
        if self._zero_cond is not None:
            if cond is None:
                cond = self._zero_cond
            else:
                cond = cond.mul(0.0).add(self._zero_cond)
        elif cond is None:
            if self.allow_missing_cond:
                cond = tensor
            else:
                raise RuntimeError("TensorRT ONNX export requires a condition tensor.")
        if self.agcm_only:
            agcm = getattr(self.model, "AGCM", None)
            if agcm is None and hasattr(self.model, "base"):
                agcm = getattr(getattr(self.model, "base"), "AGCM", None)
            if agcm is not None:
                out = agcm((tensor, cond))
                if isinstance(out, (tuple, list)):
                    return out[0]
                return out
        if self.flat_model:
            out = self.model(tensor, cond)
        else:
            out = self.model((tensor, cond))
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _patch_tensorrt_qdq_casts(onnx_path: str) -> None:
    """Remove Cast nodes that separate Q/DQ from TensorRT INT8 Conv/Gemm patterns."""
    try:
        import onnx
    except Exception as exc:
        print(f"TensorRT Q/DQ Cast cleanup skipped: onnx import failed ({exc})")
        return

    try:
        model = onnx.load(onnx_path)
    except Exception as exc:
        print(f"TensorRT Q/DQ Cast cleanup skipped: {exc}")
        return

    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}

    consumers = {}
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            consumers.setdefault(input_name, []).append((node, index))

    qdq_to_conv = 0
    conv_to_qdq = 0
    removable = set()
    compute_ops = {"Conv", "Gemm", "MatMul"}

    for node in graph.node:
        if node.op_type != "Cast" or not node.input or not node.output:
            continue

        cast_input = node.input[0]
        cast_output = node.output[0]
        producer = producers.get(cast_input)
        node_consumers = list(consumers.get(cast_output, ()))

        rewired = 0
        if producer is not None and producer.op_type == "DequantizeLinear":
            for consumer, index in node_consumers:
                if consumer.op_type in compute_ops:
                    consumer.input[index] = cast_input
                    rewired += 1
                    qdq_to_conv += 1
        else:
            for consumer, index in node_consumers:
                if consumer.op_type == "QuantizeLinear" and index == 0:
                    consumer.input[index] = cast_input
                    rewired += 1
                    conv_to_qdq += 1

        if rewired == len(node_consumers) and rewired:
            removable.add(node.name or cast_output)

    if removable:
        kept_nodes = [
            node for node in graph.node
            if not (node.op_type == "Cast" and (node.name or node.output[0]) in removable)
        ]
        del graph.node[:]
        graph.node.extend(kept_nodes)

    if qdq_to_conv or conv_to_qdq:
        data_path = f"{onnx_path}.data"
        try:
            if os.path.isfile(data_path):
                os.remove(data_path)
        except Exception:
            pass
        try:
            onnx.checker.check_model(model)
        except Exception as exc:
            print(f"TensorRT Q/DQ Cast cleanup check warning: {exc}")
        onnx.save_model(
            model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(data_path),
            size_threshold=1024,
        )
        print(
            "TensorRT Q/DQ Cast cleanup: "
            f"{qdq_to_conv} DQ->compute, {conv_to_qdq} compute->Q Cast(s) removed"
        )


def _patch_tensorrt_dq_zero_points(onnx_path: str) -> None:
    """Drop optional all-zero DQ zero-points that confuse TensorRT scale typing."""
    try:
        import onnx
        from onnx import numpy_helper
    except Exception as exc:
        print(f"TensorRT DQ zero-point cleanup skipped: onnx import failed ({exc})")
        return

    try:
        model = onnx.load(onnx_path, load_external_data=True)
    except Exception as exc:
        print(f"TensorRT DQ zero-point cleanup skipped: {exc}")
        return

    initializers = {init.name: init for init in model.graph.initializer}
    removed = 0
    skipped_nonzero = 0
    skipped_dynamic = 0
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 3:
            continue
        zero_name = node.input[2]
        initializer = initializers.get(zero_name)
        if initializer is None:
            skipped_dynamic += 1
            continue
        try:
            array = numpy_helper.to_array(initializer)
        except Exception:
            skipped_dynamic += 1
            continue
        if array.size and not bool(np.all(array == 0)):
            skipped_nonzero += 1
            continue
        del node.input[2:]
        removed += 1

    if not removed:
        print(
            "TensorRT DQ zero-point cleanup: no removable all-zero "
            f"zero-points found (dynamic={skipped_dynamic}, nonzero={skipped_nonzero})"
        )
        return

    data_path = f"{onnx_path}.data"
    try:
        if os.path.isfile(data_path):
            os.remove(data_path)
    except Exception:
        pass
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"TensorRT DQ zero-point cleanup check warning: {exc}")
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
        size_threshold=1024,
    )
    print(
        "TensorRT DQ zero-point cleanup: "
        f"removed {removed} all-zero DQ zero-point input(s)"
    )


def _patch_tensorrt_qdq_zero_points(onnx_path: str) -> None:
    """Drop optional all-zero Q/DQ zero-points from explicit quantized graphs."""
    try:
        import onnx
        from onnx import numpy_helper
    except Exception as exc:
        print(f"TensorRT Q/DQ zero-point cleanup skipped: onnx import failed ({exc})")
        return

    try:
        model = onnx.load(onnx_path, load_external_data=True)
    except Exception as exc:
        print(f"TensorRT Q/DQ zero-point cleanup skipped: {exc}")
        return

    initializers = {init.name: init for init in model.graph.initializer}
    removed = 0
    skipped_nonzero = 0
    skipped_dynamic = 0
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if len(node.input) < 3:
            continue
        zero_name = node.input[2]
        initializer = initializers.get(zero_name)
        if initializer is None:
            skipped_dynamic += 1
            continue
        try:
            array = numpy_helper.to_array(initializer)
        except Exception:
            skipped_dynamic += 1
            continue
        if array.size and not bool(np.all(array == 0)):
            skipped_nonzero += 1
            continue
        del node.input[2:]
        removed += 1

    if not removed:
        print(
            "TensorRT Q/DQ zero-point cleanup: no removable all-zero "
            f"zero-points found (dynamic={skipped_dynamic}, nonzero={skipped_nonzero})"
        )
        return

    data_path = f"{onnx_path}.data"
    try:
        if os.path.isfile(data_path):
            os.remove(data_path)
    except Exception:
        pass
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"TensorRT Q/DQ zero-point cleanup check warning: {exc}")
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
        size_threshold=1024,
    )
    print(
        "TensorRT Q/DQ zero-point cleanup: "
        f"removed {removed} all-zero Q/DQ zero-point input(s)"
    )


def _patch_tensorrt_elementwise_input_qdq(
    onnx_path: str,
    op_types: tuple[str, ...] = ("Add", "Mul"),
) -> None:
    """Add Q/DQ on elementwise inputs that already feed calibrated Q nodes."""
    try:
        import onnx
    except Exception as exc:
        print(f"TensorRT elementwise Q/DQ fusion skipped: onnx import failed ({exc})")
        return

    try:
        model = onnx.load(onnx_path)
    except Exception as exc:
        print(f"TensorRT elementwise Q/DQ fusion skipped: {exc}")
        return

    graph = model.graph
    enabled_ops = {str(op) for op in op_types if str(op)}
    producers = {output: node for node in graph.node for output in node.output}
    initializers = {init.name for init in graph.initializer}
    graph_inputs = {value.name for value in graph.input}

    consumers = {}
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            consumers.setdefault(input_name, []).append((node, index))

    existing_names = {node.name for node in graph.node if node.name}
    existing_tensors = set()
    for node in graph.node:
        existing_tensors.update(node.input)
        existing_tensors.update(node.output)

    def _unique(base: str, used: set[str]) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_") or "tensor"
        name = safe
        index = 1
        while name in used:
            name = f"{safe}_{index}"
            index += 1
        used.add(name)
        return name

    def _output_quantizer(elementwise_node) -> object | None:
        for consumer, index in consumers.get(elementwise_node.output[0], []):
            if consumer.op_type == "QuantizeLinear" and index == 0:
                return consumer
        return None

    patched_by_op = {op: 0 for op in enabled_ops}
    inserted_qdq = 0
    new_nodes = []

    for node in graph.node:
        if node.op_type not in enabled_ops:
            new_nodes.append(node)
            continue

        output_quant = _output_quantizer(node)
        if output_quant is None or len(output_quant.input) < 2:
            new_nodes.append(node)
            continue

        scale_name = output_quant.input[1]
        zero_name = output_quant.input[2] if len(output_quant.input) > 2 else ""
        node_was_patched = False

        for input_index, input_name in enumerate(list(node.input)):
            producer = producers.get(input_name)
            if input_name in initializers or input_name in graph_inputs:
                continue
            if producer is not None and producer.op_type == "DequantizeLinear":
                continue

            q_name = _unique(f"{node.name or node.output[0]}_input{input_index}_QuantizeLinear", existing_names)
            dq_name = _unique(f"{node.name or node.output[0]}_input{input_index}_DequantizeLinear", existing_names)
            q_out = _unique(f"{node.output[0]}_input{input_index}_q", existing_tensors)
            dq_out = _unique(f"{node.output[0]}_input{input_index}_dq", existing_tensors)
            q_inputs = [input_name, scale_name]
            dq_inputs = [q_out, scale_name]
            if zero_name:
                q_inputs.append(zero_name)
                dq_inputs.append(zero_name)

            new_nodes.append(
                onnx.helper.make_node(
                    "QuantizeLinear",
                    q_inputs,
                    [q_out],
                    name=q_name,
                )
            )
            new_nodes.append(
                onnx.helper.make_node(
                    "DequantizeLinear",
                    dq_inputs,
                    [dq_out],
                    name=dq_name,
                )
            )
            node.input[input_index] = dq_out
            inserted_qdq += 1
            node_was_patched = True

        if node_was_patched:
            patched_by_op[node.op_type] = patched_by_op.get(node.op_type, 0) + 1
        new_nodes.append(node)

    patched_total = sum(patched_by_op.values())
    label = "+".join(sorted(enabled_ops)) or "elementwise"
    if not patched_total:
        print(f"TensorRT {label} Q/DQ fusion: no eligible nodes found")
        return

    del graph.node[:]
    graph.node.extend(new_nodes)

    data_path = f"{onnx_path}.data"
    try:
        if os.path.isfile(data_path):
            os.remove(data_path)
    except Exception:
        pass
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"TensorRT {label} Q/DQ fusion check warning: {exc}")
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
        size_threshold=1024,
    )
    patched_summary = ", ".join(
        f"{op}={count}" for op, count in sorted(patched_by_op.items()) if count
    )
    print(
        f"TensorRT {label} Q/DQ fusion: "
        f"patched {patched_summary}, inserted {inserted_qdq} input Q/DQ pair(s)"
    )


def _patch_tensorrt_add_input_qdq(onnx_path: str) -> None:
    _patch_tensorrt_elementwise_input_qdq(onnx_path, ("Add",))


def _patch_tensorrt_mul_input_qdq(onnx_path: str) -> None:
    """Add Q/DQ on Mul inputs that feed already-calibrated Q/DQ regions."""
    try:
        import onnx
    except Exception as exc:
        print(f"TensorRT Mul Q/DQ fusion skipped: onnx import failed ({exc})")
        return

    try:
        model = onnx.load(onnx_path)
    except Exception as exc:
        print(f"TensorRT Mul Q/DQ fusion skipped: {exc}")
        return

    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    initializers = {init.name for init in graph.initializer}
    graph_inputs = {value.name for value in graph.input}

    consumers = {}
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            consumers.setdefault(input_name, []).append((node, index))

    existing_names = {node.name for node in graph.node if node.name}
    existing_tensors = set()
    for node in graph.node:
        existing_tensors.update(node.input)
        existing_tensors.update(node.output)

    def _unique(base: str, used: set[str]) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_") or "tensor"
        name = safe
        index = 1
        while name in used:
            name = f"{safe}_{index}"
            index += 1
        used.add(name)
        return name

    def _output_quantizer(node) -> object | None:
        for consumer, index in consumers.get(node.output[0], []):
            if consumer.op_type == "QuantizeLinear" and index == 0:
                return consumer
        return None

    def _mul_quantizer(node) -> object | None:
        direct = _output_quantizer(node)
        if direct is not None:
            return direct
        for consumer, _index in consumers.get(node.output[0], []):
            if consumer.op_type == "Add":
                add_quant = _output_quantizer(consumer)
                if add_quant is not None:
                    return add_quant
        return None

    patched_muls = 0
    inserted_qdq = 0
    new_nodes = []

    for node in graph.node:
        if node.op_type != "Mul":
            new_nodes.append(node)
            continue

        output_quant = _mul_quantizer(node)
        if output_quant is None or len(output_quant.input) < 2:
            new_nodes.append(node)
            continue

        scale_name = output_quant.input[1]
        zero_name = output_quant.input[2] if len(output_quant.input) > 2 else ""
        node_was_patched = False

        for input_index, input_name in enumerate(list(node.input)):
            producer = producers.get(input_name)
            if input_name in initializers or input_name in graph_inputs:
                continue
            if producer is not None and producer.op_type == "DequantizeLinear":
                continue

            q_name = _unique(f"{node.name or node.output[0]}_input{input_index}_QuantizeLinear", existing_names)
            dq_name = _unique(f"{node.name or node.output[0]}_input{input_index}_DequantizeLinear", existing_names)
            q_out = _unique(f"{node.output[0]}_input{input_index}_q", existing_tensors)
            dq_out = _unique(f"{node.output[0]}_input{input_index}_dq", existing_tensors)
            q_inputs = [input_name, scale_name]
            dq_inputs = [q_out, scale_name]
            if zero_name:
                q_inputs.append(zero_name)
                dq_inputs.append(zero_name)

            new_nodes.append(
                onnx.helper.make_node(
                    "QuantizeLinear",
                    q_inputs,
                    [q_out],
                    name=q_name,
                )
            )
            new_nodes.append(
                onnx.helper.make_node(
                    "DequantizeLinear",
                    dq_inputs,
                    [dq_out],
                    name=dq_name,
                )
            )
            node.input[input_index] = dq_out
            inserted_qdq += 1
            node_was_patched = True

        if node_was_patched:
            patched_muls += 1
        new_nodes.append(node)

    if not patched_muls:
        print("TensorRT Mul Q/DQ fusion: no eligible Mul nodes found")
        return

    del graph.node[:]
    graph.node.extend(new_nodes)

    data_path = f"{onnx_path}.data"
    try:
        if os.path.isfile(data_path):
            os.remove(data_path)
    except Exception:
        pass
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"TensorRT Mul Q/DQ fusion check warning: {exc}")
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
        size_threshold=1024,
    )
    print(
        "TensorRT Mul Q/DQ fusion: "
        f"patched {patched_muls} Mul node(s), inserted {inserted_qdq} input Q/DQ pair(s)"
    )


def _patch_tensorrt_fp16_constants(onnx_path: str) -> None:
    """Convert safe FP32 ONNX constants to FP16 for TensorRT mixed INT8/FP16 engines."""
    try:
        import onnx
        from onnx import TensorProto, numpy_helper
    except Exception as exc:
        print(f"TensorRT FP16 constant cleanup skipped: onnx import failed ({exc})")
        return

    try:
        model = onnx.load(onnx_path)
    except Exception as exc:
        print(f"TensorRT FP16 constant cleanup skipped: {exc}")
        return

    graph = model.graph
    protected = set()
    graph_io = {value.name for value in graph.input}
    graph_io.update(value.name for value in graph.output)

    for node in graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            # Keep calibration scales and zero-points exactly as exported.
            protected.update(node.input[1:])

    changed_initializers = 0
    for initializer in graph.initializer:
        if (
            initializer.data_type != TensorProto.FLOAT
            or initializer.name in protected
            or initializer.name in graph_io
        ):
            continue
        array = numpy_helper.to_array(initializer)
        if array.dtype != np.float32:
            continue
        fp16_tensor = numpy_helper.from_array(array.astype(np.float16), initializer.name)
        initializer.CopyFrom(fp16_tensor)
        changed_initializers += 1

    changed_constants = 0
    for node in graph.node:
        if node.op_type != "Constant" or any(output in protected for output in node.output):
            continue
        for attr in node.attribute:
            if attr.name != "value" or not attr.HasField("t"):
                continue
            tensor = attr.t
            if tensor.data_type != TensorProto.FLOAT:
                continue
            array = numpy_helper.to_array(tensor)
            if array.dtype != np.float32:
                continue
            attr.t.CopyFrom(numpy_helper.from_array(array.astype(np.float16), tensor.name))
            changed_constants += 1

    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[object]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    def _constant_dtype(name: str):
        for initializer in graph.initializer:
            if initializer.name == name:
                return initializer.data_type
        producer = producers.get(name)
        if producer is None or producer.op_type != "Constant":
            return None
        for attr in producer.attribute:
            if attr.name == "value" and attr.HasField("t"):
                return attr.t.data_type
        return None

    changed_casts = 0
    compare_ops = {"Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal"}
    for node in graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        to_attr = next((attr for attr in node.attribute if attr.name == "to"), None)
        if to_attr is None or to_attr.i != TensorProto.FLOAT:
            continue
        for consumer in consumers.get(node.output[0], []):
            if consumer.op_type not in compare_ops:
                continue
            if any(
                input_name != node.output[0]
                and _constant_dtype(input_name) == TensorProto.FLOAT16
                for input_name in consumer.input
            ):
                to_attr.i = TensorProto.FLOAT16
                changed_casts += 1
                break

    if not (changed_initializers or changed_constants or changed_casts):
        print("TensorRT FP16 constant cleanup: no safe FP32 constants found")
        return

    data_path = f"{onnx_path}.data"
    try:
        if os.path.isfile(data_path):
            os.remove(data_path)
    except Exception:
        pass
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"TensorRT FP16 constant cleanup check warning: {exc}")
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
        size_threshold=1024,
    )
    print(
        "TensorRT FP16 constant cleanup: "
        f"{changed_initializers} initializer(s), {changed_constants} Constant node(s), "
        f"{changed_casts} Cast node(s)"
    )


def _patch_tensorrt_fp16_qdq_scales(onnx_path: str) -> None:
    """Keep Q/DQ scales in FP32 so TensorRT does not quantize scale layers."""
    try:
        import onnx
        from onnx import TensorProto, numpy_helper
        import numpy as np
    except Exception as exc:
        print(f"TensorRT FP32 Q/DQ scale cleanup skipped: onnx import failed ({exc})")
        return

    try:
        model = onnx.load(onnx_path, load_external_data=True)
    except Exception as exc:
        print(f"TensorRT FP32 Q/DQ scale cleanup skipped: {exc}")
        return

    scale_names = set()
    for node in model.graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"} and len(node.input) > 1:
            scale_names.add(node.input[1])

    changed = 0
    for initializer in model.graph.initializer:
        if initializer.name not in scale_names:
            continue
        array = numpy_helper.to_array(initializer)
        if array.dtype == np.float32 and initializer.data_type == TensorProto.FLOAT:
            continue
        initializer.CopyFrom(
            numpy_helper.from_array(array.astype(np.float32), initializer.name)
        )
        changed += 1

    if not changed:
        print("TensorRT FP32 Q/DQ scale cleanup: scales already FP32")
        return

    for opset in model.opset_import:
        if not opset.domain and opset.version < 19:
            opset.version = 19

    data_path = f"{onnx_path}.data"
    try:
        if os.path.isfile(data_path):
            os.remove(data_path)
    except Exception:
        pass
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"TensorRT FP32 Q/DQ scale cleanup check warning: {exc}")
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
        size_threshold=1024,
    )
    print(f"TensorRT FP32 Q/DQ scale cleanup: {changed} scale initializer(s)")


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
    qdq_fusion: str = "auto",
    int8_zero_condition: bool = False,
    int8_single_input: bool = False,
    int8_agcm_only: bool = False,
    single_input_graph: bool = False,
) -> str:
    if os.path.isfile(onnx_path):
        try:
            os.remove(onnx_path)
            data_path = f"{onnx_path}.data"
            if os.path.isfile(data_path):
                os.remove(data_path)
            print(f"TensorRT stale ONNX removed before export: {onnx_path}")
        except Exception as exc:
            print(f"TensorRT stale ONNX removal skipped: {exc}")

    export_model = getattr(model, "_orig_mod", model).eval()
    int8_export = str(precision).startswith("int8")
    fusion = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
    native_int8_export = int8_export and fusion == "native"
    if int8_export:
        if native_int8_export:
            export_model = _convert_model_to_tensorrt_native_layers(export_model).eval()
        else:
            export_model = _convert_model_to_tensorrt_qdq(export_model).eval()
    _patch_adaptive_avgpool_for_trt(export_model)
    h, w = int(height), int(width)
    cond_h, cond_w = max(1, h // 4), max(1, w // 4)
    tensor = torch.zeros((1, 3, h, w), dtype=dtype, device=device)
    cond = torch.zeros((1, 3, cond_h, cond_w), dtype=dtype, device=device)
    zero_cond = (
        torch.zeros_like(cond)
        if (int8_export and bool(int8_zero_condition))
        else None
    )
    single_input = bool(
        zero_cond is not None and bool(int8_single_input)
    ) or bool(single_input_graph)
    agcm_only = bool(int8_agcm_only)
    if zero_cond is not None:
        print("TensorRT INT8 ONNX export: constant zero condition enabled")
    if single_input:
        print("TensorRT ONNX export: single input enabled")
    if agcm_only:
        print("TensorRT INT8 ONNX export: AGCM-only speed graph enabled")
    wrapper = _ONNXExportWrapper(
        export_model,
        flat_model=flat_model,
        zero_cond=zero_cond,
        agcm_only=agcm_only,
        allow_missing_cond=single_input,
    ).eval()
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    print(f"TensorRT ONNX export: {onnx_path}")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    with torch.inference_mode():
        export_args = (tensor,) if single_input else (tensor, cond)
        input_names = ["input"] if single_input else ["input", "cond"]
        torch.onnx.export(
            wrapper,
            export_args,
            onnx_path,
            verbose=False,
            input_names=input_names,
            output_names=["output"],
            opset_version=19 if (int8_export and not native_int8_export) else 18,
            do_constant_folding=True,
        )
    if int8_export and native_int8_export:
        print("TensorRT native INT8 export: Q/DQ graph patches skipped")
    elif int8_export:
        _patch_tensorrt_qdq_casts(onnx_path)
        _patch_tensorrt_qdq_zero_points(onnx_path)
        if fusion == "add":
            _patch_tensorrt_add_input_qdq(onnx_path)
        elif fusion == "add-mul":
            _patch_tensorrt_elementwise_input_qdq(onnx_path, ("Add", "Mul"))
        if dtype == torch.float16:
            _patch_tensorrt_fp16_qdq_scales(onnx_path)
            _patch_tensorrt_fp16_constants(onnx_path)
        else:
            print(
                "TensorRT FP16 ONNX cleanup skipped: "
                f"export dtype is {dtype}, keeping FP32 graph valid for validation"
            )
    return onnx_path


def _remove_onnx_artifacts(path: str) -> None:
    for candidate in (path, f"{path}.data"):
        try:
            if candidate and os.path.isfile(candidate):
                os.remove(candidate)
        except Exception:
            pass


def _unwrap_model_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _make_tensorrt_modelopt_torch_batches(
    *,
    model: nn.Module,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
    steps: int,
    include_targets: bool,
) -> list[tuple[torch.Tensor, ...]]:
    h, w = int(height), int(width)
    cond_h, cond_w = max(1, h // 4), max(1, w // 4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_tensorrt_int8_modelopt_torch_seed())
    batches: list[list[torch.Tensor]] = []
    for index in range(max(1, int(steps))):
        base = torch.rand((1, 3, h, w), generator=generator, dtype=torch.float32)
        if index % 5 == 1:
            base = base * 0.55
        elif index % 5 == 2:
            base = torch.sqrt(base.clamp_min(0.0))
        elif index % 5 == 3:
            ramp = torch.linspace(0.0, 1.0, w, dtype=torch.float32).view(1, 1, 1, w)
            base = (base * 0.65) + (ramp * 0.35)
        elif index % 5 == 4:
            ramp = torch.linspace(0.0, 1.0, h, dtype=torch.float32).view(1, 1, h, 1)
            base = (base * 0.65) + (ramp * 0.35)
        tensor = base.clamp_(0.0, 1.0).to(device=device, dtype=dtype).contiguous()
        cond = F.interpolate(
            tensor,
            size=(cond_h, cond_w),
            mode="bilinear",
            align_corners=False,
        ).contiguous()
        batches.append([tensor, cond])

    if include_targets:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in batches:
                target = _unwrap_model_output(model((batch[0], batch[1]))).detach()
                batch.append(target.to(device=device, dtype=dtype).contiguous())
        model.train(was_training)
    return [tuple(batch) for batch in batches]


def _tensorrt_modelopt_torch_forward_step(
    model: nn.Module,
    batch,
) -> torch.Tensor:
    tensor = batch[0]
    cond = batch[1]
    return _unwrap_model_output(model((tensor, cond)))


def _count_tensorrt_modelopt_torch_quantizers(model: nn.Module) -> tuple[int, int]:
    enabled = 0
    total = 0
    for module in model.modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        total += 1
        state = getattr(module, "is_enabled", None)
        try:
            is_enabled = bool(state() if callable(state) else state)
        except Exception:
            is_enabled = bool(getattr(module, "_is_enabled", False))
        enabled += int(is_enabled)
    return enabled, total


def _tensorrt_modelopt_torch_quantizer_kind(name: str) -> str:
    for suffix in (
        "input_quantizer",
        "weight_quantizer",
        "output_quantizer",
        "_input_quantizer",
        "_weight_quantizer",
        "_output_quantizer",
    ):
        if str(name).endswith(suffix):
            return suffix.lstrip("_")
    return "other"


def _summarize_tensorrt_modelopt_torch_quantizers(
    model: nn.Module,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        kind = _tensorrt_modelopt_torch_quantizer_kind(name)
        entry = summary.setdefault(kind, {"enabled": 0, "total": 0, "disabled": []})
        enabled = _tensorrt_modelopt_torch_quantizer_enabled(module)
        entry["total"] = int(entry["total"]) + 1
        entry["enabled"] = int(entry["enabled"]) + int(enabled)
        if not enabled:
            disabled = entry["disabled"]
            if isinstance(disabled, list):
                disabled.append(name)
    return summary


def _print_tensorrt_modelopt_torch_quantizer_summary(model: nn.Module) -> None:
    summary = _summarize_tensorrt_modelopt_torch_quantizers(model)
    for kind in ("input_quantizer", "weight_quantizer", "output_quantizer", "other"):
        entry = summary.get(kind)
        if not entry:
            continue
        print(
            "TensorRT INT8 ModelOpt Torch "
            f"{kind}: {int(entry['enabled'])}/{int(entry['total'])} enabled"
        )


def _enable_tensorrt_modelopt_torch_output_quantizers(model: nn.Module) -> int:
    changed = 0
    failed: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        if _tensorrt_modelopt_torch_quantizer_kind(name) != "output_quantizer":
            continue
        if _tensorrt_modelopt_torch_quantizer_enabled(module):
            continue
        try:
            if hasattr(module, "enable"):
                module.enable()
            else:
                setattr(module, "_disabled", False)
                setattr(module, "_is_enabled", True)
        except Exception:
            failed.append(name)
            continue
        if _tensorrt_modelopt_torch_quantizer_enabled(module):
            changed += 1
        else:
            failed.append(name)
    if failed:
        preview = ", ".join(failed[:12])
        more = "" if len(failed) <= 12 else f", +{len(failed) - 12} more"
        raise RuntimeError(
            "TensorRT INT8 Full contract failed: could not enable "
            f"{len(failed)} output quantizer(s) ({preview}{more})."
        )
    if changed:
        print(
            "TensorRT INT8 Full contract: enabled "
            f"{changed} output quantizer(s)"
        )
    return changed


def _enable_tensorrt_modelopt_torch_quantizers(
    model: nn.Module,
    *,
    kinds: tuple[str, ...],
    label: str,
) -> int:
    changed = 0
    failed: list[str] = []
    wanted = set(kinds)
    for name, module in model.named_modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        if _tensorrt_modelopt_torch_quantizer_kind(name) not in wanted:
            continue
        if _tensorrt_modelopt_torch_quantizer_enabled(module):
            continue
        try:
            if hasattr(module, "enable"):
                module.enable()
            else:
                setattr(module, "_disabled", False)
                setattr(module, "_is_enabled", True)
        except Exception:
            failed.append(name)
            continue
        if _tensorrt_modelopt_torch_quantizer_enabled(module):
            changed += 1
        else:
            failed.append(name)
    if failed:
        preview = ", ".join(failed[:12])
        more = "" if len(failed) <= 12 else f", +{len(failed) - 12} more"
        raise RuntimeError(
            f"TensorRT INT8 {label} contract failed: could not enable "
            f"{len(failed)} quantizer(s) ({preview}{more})."
        )
    if changed:
        print(
            f"TensorRT INT8 {label} contract: enabled "
            f"{changed} quantizer(s)"
        )
    return changed


def _enable_tensorrt_modelopt_torch_filtered_output_quantizers(model: nn.Module) -> int:
    include = _tensorrt_int8_modelopt_torch_include_patterns()
    exclude = _tensorrt_int8_modelopt_torch_exclude_patterns()
    changed = 0
    failed: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        if _tensorrt_modelopt_torch_quantizer_kind(name) != "output_quantizer":
            continue
        parent_name = _tensorrt_modelopt_torch_quantizer_parent(name)
        keep = True
        if include:
            keep = _tensorrt_modelopt_torch_pattern_match(name, parent_name, include)
        if keep and exclude:
            keep = not _tensorrt_modelopt_torch_pattern_match(
                name,
                parent_name,
                exclude,
            )
        if not keep or _tensorrt_modelopt_torch_quantizer_enabled(module):
            continue
        try:
            if hasattr(module, "enable"):
                module.enable()
            else:
                setattr(module, "_disabled", False)
                setattr(module, "_is_enabled", True)
        except Exception:
            failed.append(name)
            continue
        if _tensorrt_modelopt_torch_quantizer_enabled(module):
            changed += 1
        else:
            failed.append(name)
    if failed:
        preview = ", ".join(failed[:12])
        more = "" if len(failed) <= 12 else f", +{len(failed) - 12} more"
        raise RuntimeError(
            "TensorRT INT8 mixed output quantizer request failed: could not "
            f"enable {len(failed)} output quantizer(s) ({preview}{more})."
        )
    if changed:
        print(
            "TensorRT INT8 ModelOpt Torch mixed outputs: enabled "
            f"{changed} included output quantizer(s)"
        )
    return changed


def _calibrate_tensorrt_modelopt_torch_enabled_output_quantizers(
    model: nn.Module,
    forward_loop,
) -> int:
    return _calibrate_tensorrt_modelopt_torch_enabled_activation_quantizers(
        model,
        forward_loop,
        kinds=("output_quantizer",),
        label="mixed output",
    )


def _calibrate_tensorrt_modelopt_torch_enabled_activation_quantizers(
    model: nn.Module,
    forward_loop,
    *,
    kinds: tuple[str, ...],
    label: str,
) -> int:
    targets = []
    wanted = set(kinds)
    for name, module in model.named_modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        if _tensorrt_modelopt_torch_quantizer_kind(name) not in wanted:
            continue
        if not _tensorrt_modelopt_torch_quantizer_enabled(module):
            continue
        amax = getattr(module, "_amax", None)
        if amax is not None:
            continue
        targets.append((name, module))
    if not targets:
        return 0

    armed = []
    for _name, module in targets:
        try:
            if hasattr(module, "reset_amax"):
                module.reset_amax()
            if hasattr(module, "disable_quant"):
                module.disable_quant()
            if hasattr(module, "enable_calib"):
                module.enable_calib()
            armed.append(module)
        except Exception:
            continue
    if not armed:
        return 0

    with _tensorrt_modelopt_torch_no_inplace_functionals():
        forward_loop(model)

    calibrated = 0
    failed: list[str] = []
    armed_ids = {id(module) for module in armed}
    for name, module in targets:
        if id(module) not in armed_ids:
            continue
        try:
            if hasattr(module, "disable_calib"):
                module.disable_calib()
            if hasattr(module, "load_calib_amax"):
                module.load_calib_amax(strict=True)
            if hasattr(module, "enable_quant"):
                module.enable_quant()
        except Exception:
            failed.append(name)
            try:
                if hasattr(module, "disable_calib"):
                    module.disable_calib()
                if hasattr(module, "enable_quant"):
                    module.enable_quant()
            except Exception:
                pass
            continue
        if hasattr(module, "_amax"):
            calibrated += 1
        else:
            failed.append(name)

    if failed:
        preview = ", ".join(failed[:12])
        more = "" if len(failed) <= 12 else f", +{len(failed) - 12} more"
        raise RuntimeError(
            f"TensorRT INT8 {label} quantizer calibration failed for "
            f"{len(failed)} quantizer(s) ({preview}{more})."
        )
    if calibrated:
        print(
            f"TensorRT INT8 ModelOpt Torch {label}: calibrated "
            f"{calibrated} quantizer(s)"
        )
    return calibrated


def _enforce_tensorrt_modelopt_torch_full_quantizers(model: nn.Module) -> None:
    summary = _summarize_tensorrt_modelopt_torch_quantizers(model)
    disabled_compute: list[str] = []
    for kind in ("input_quantizer", "weight_quantizer", "output_quantizer"):
        entry = summary.get(kind, {})
        disabled = entry.get("disabled", [])
        if isinstance(disabled, list):
            disabled_compute.extend(str(name) for name in disabled)
    if not disabled_compute:
        return
    preview = ", ".join(disabled_compute[:12])
    more = "" if len(disabled_compute) <= 12 else f", +{len(disabled_compute) - 12} more"
    raise RuntimeError(
        "TensorRT INT8 Full contract failed: ModelOpt left "
        f"{len(disabled_compute)} quantizer(s) disabled "
        f"({preview}{more}). Use int8-mixed for partial quantization."
    )


def _tensorrt_modelopt_torch_int8_config(mtq_module, *, full_outputs: bool = False):
    cfg = copy.deepcopy(mtq_module.INT8_DEFAULT_CFG)
    if full_outputs:
        quant_cfg = cfg.setdefault("quant_cfg", [])
        if isinstance(quant_cfg, list):
            quant_cfg.insert(
                3,
                {
                    "quantizer_name": "*output_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                },
            )
    quant_scheme = _tensorrt_int8_modelopt_torch_quant_scheme()
    for entry in cfg.get("quant_cfg", []):
        if not isinstance(entry, dict):
            continue
        quantizer_name = str(entry.get("quantizer_name", ""))
        qcfg = entry.get("cfg")
        if not isinstance(qcfg, dict):
            continue
        if quantizer_name.endswith("weight_quantizer"):
            qcfg["unsigned"] = False
            qcfg["narrow_range"] = False
        elif quantizer_name.endswith("input_quantizer") or quantizer_name.endswith("output_quantizer"):
            qcfg["unsigned"] = quant_scheme == "asymmetric"
            qcfg["narrow_range"] = False
    return cfg


def _tensorrt_modelopt_torch_quantizer_enabled(module: nn.Module) -> bool:
    state = getattr(module, "is_enabled", None)
    try:
        return bool(state() if callable(state) else state)
    except Exception:
        return not bool(getattr(module, "_disabled", False))


def _tensorrt_modelopt_torch_quantizer_parent(name: str) -> str:
    markers = (
        ".input_quantizer",
        ".weight_quantizer",
        ".output_quantizer",
        "._input_quantizer",
        "._weight_quantizer",
        "._output_quantizer",
    )
    for marker in markers:
        index = name.find(marker)
        if index >= 0:
            return name[:index]
    parts = name.split(".")
    if parts and "quantizer" in parts[-1].lower():
        return ".".join(parts[:-1])
    return name


def _tensorrt_modelopt_torch_pattern_match(
    name: str,
    parent_name: str,
    patterns: tuple[str, ...],
) -> bool:
    if not patterns:
        return False
    candidates = (name, parent_name)
    for raw_pattern in patterns:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        pattern_lower = pattern.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if (
                pattern_lower == candidate_lower
                or candidate_lower.startswith(pattern_lower)
                or pattern_lower in candidate_lower
            ):
                return True
        try:
            if any(re.search(pattern, candidate) for candidate in candidates):
                return True
        except re.error:
            pass
    return False


def _apply_tensorrt_modelopt_torch_quantizer_filters(
    model: nn.Module,
    precision: str | None = None,
) -> tuple[int, int, int]:
    include = _tensorrt_int8_modelopt_torch_include_patterns()
    exclude = _tensorrt_int8_modelopt_torch_exclude_patterns()
    has_hg = any(name == "hg" or name.startswith("hg.") for name, _ in model.named_modules())
    if (
        has_hg
        and not include
        and not exclude
        and _tensorrt_int8_modelopt_torch_hg_default_enabled()
    ):
        include = ("hg",)
        if str(precision or "").startswith("int8-mixed"):
            exclude = _tensorrt_int8_modelopt_torch_hg_mixed_exclude_patterns()

    if not include and not exclude:
        enabled, total = _count_tensorrt_modelopt_torch_quantizers(model)
        return enabled, total, 0

    total = 0
    enabled = 0
    changed = 0
    enabled_parents: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        total += 1
        parent_name = _tensorrt_modelopt_torch_quantizer_parent(name)
        keep = True
        if include:
            keep = _tensorrt_modelopt_torch_pattern_match(name, parent_name, include)
        if keep and exclude:
            keep = not _tensorrt_modelopt_torch_pattern_match(
                name,
                parent_name,
                exclude,
            )
        was_enabled = _tensorrt_modelopt_torch_quantizer_enabled(module)
        try:
            if keep:
                pass
            elif hasattr(module, "disable"):
                module.disable()
            else:
                setattr(module, "_disabled", True)
        except Exception:
            pass
        is_enabled = _tensorrt_modelopt_torch_quantizer_enabled(module)
        enabled += int(is_enabled)
        changed += int(was_enabled != is_enabled)
        if is_enabled and parent_name not in enabled_parents:
            enabled_parents.append(parent_name)

    print(
        "TensorRT INT8 ModelOpt Torch quantizer filter: "
        f"include={list(include) or '*'}, exclude={list(exclude) or 'none'}, "
        f"enabled={enabled}/{total}, changed={changed}"
    )
    if enabled_parents:
        preview = ", ".join(enabled_parents[:12])
        more = "" if len(enabled_parents) <= 12 else f", +{len(enabled_parents) - 12} more"
        print(f"TensorRT INT8 ModelOpt Torch enabled blocks: {preview}{more}")
    return enabled, total, changed


@contextlib.contextmanager
def _tensorrt_modelopt_torch_no_inplace_functionals():
    original_relu = F.relu
    original_leaky_relu = F.leaky_relu

    def _relu_no_inplace(input, inplace: bool = False):
        return original_relu(input, inplace=False)

    def _leaky_relu_no_inplace(input, negative_slope: float = 0.01, inplace: bool = False):
        return original_leaky_relu(input, negative_slope=negative_slope, inplace=False)

    F.relu = _relu_no_inplace
    F.leaky_relu = _leaky_relu_no_inplace
    try:
        yield
    finally:
        F.relu = original_relu
        F.leaky_relu = original_leaky_relu


def _disable_tensorrt_modelopt_torch_inplace_activations(model: nn.Module) -> int:
    changed = 0
    for module in model.modules():
        if not hasattr(module, "inplace"):
            continue
        try:
            if bool(getattr(module, "inplace")):
                setattr(module, "inplace", False)
                changed += 1
        except Exception:
            pass
    if changed:
        print(
            "TensorRT INT8 ModelOpt Torch auto_quantize: "
            f"disabled {changed} in-place activation module(s) for scoring"
        )
    return changed


def _set_tensorrt_modelopt_torch_export_dtype(
    model: nn.Module,
    dtype: torch.dtype,
) -> None:
    high_precision = _tensorrt_int8_modelopt_torch_qdq_dtype(dtype)
    changed = 0
    for module in model.modules():
        if module.__class__.__name__ != "TensorQuantizer":
            continue
        try:
            module.trt_high_precision_dtype = high_precision
            changed += 1
        except Exception:
            pass
    if changed:
        print(
            "TensorRT INT8 ModelOpt Torch export dtype: "
            f"{high_precision} Q/DQ high precision on {changed} quantizer(s)"
        )


def _apply_tensorrt_modelopt_torch_int8_quantization(
    *,
    model: nn.Module,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
    precision: str,
) -> nn.Module:
    try:
        import modelopt.torch.quantization as mtq
    except Exception as exc:
        raise RuntimeError(
            "ModelOpt PyTorch quantization is not installed. "
            "Install nvidia-modelopt to use HDRTVNET_TRT_INT8_MODELOPT_TORCH=1."
        ) from exc

    mode = _tensorrt_int8_modelopt_torch_mode(precision)
    method = _tensorrt_int8_modelopt_torch_method()
    quant_scheme = _tensorrt_int8_modelopt_torch_quant_scheme()
    full_precision = str(precision or "").startswith("int8-full")
    quant_config = _tensorrt_modelopt_torch_int8_config(
        mtq,
        full_outputs=full_precision,
    )
    calib_steps = _tensorrt_int8_modelopt_torch_calib_steps()
    score_steps = _tensorrt_int8_modelopt_torch_score_steps()
    total_steps = calib_steps if mode == "full" else max(calib_steps, score_steps)
    include_targets = mode == "auto" and method == "gradient"
    _disable_tensorrt_modelopt_torch_inplace_activations(model)
    fallback_model = None
    if mode == "auto" and method == "gradient":
        try:
            fallback_model = copy.deepcopy(model).eval()
        except Exception:
            fallback_model = None
    with _tensorrt_modelopt_torch_no_inplace_functionals():
        batches = _make_tensorrt_modelopt_torch_batches(
            model=model,
            width=width,
            height=height,
            dtype=dtype,
            device=device,
            steps=total_steps,
            include_targets=include_targets,
        )

    def forward_loop(qmodel: nn.Module) -> None:
        qmodel.eval()
        with torch.no_grad():
            for batch in batches[:calib_steps]:
                _tensorrt_modelopt_torch_forward_step(qmodel, batch)

    print(
        "TensorRT INT8 ModelOpt Torch quantization: "
        f"mode={mode}, scheme={quant_scheme}, "
        f"calib_steps={calib_steps}, score_steps={score_steps}"
    )
    if mode == "full":
        with _tensorrt_modelopt_torch_no_inplace_functionals():
            quantized = mtq.quantize(
                model,
                copy.deepcopy(quant_config),
                forward_loop,
            )
    else:
        constraints = {"effective_bits": _tensorrt_int8_modelopt_torch_effective_bits()}

        def loss_func(output, batch) -> torch.Tensor:
            target = batch[2]
            return F.mse_loss(_unwrap_model_output(output).float(), target.float())

        try:
            with _tensorrt_modelopt_torch_no_inplace_functionals():
                quantized, _search_state = mtq.auto_quantize(
                    model,
                    constraints=constraints,
                    quantization_formats=[copy.deepcopy(quant_config)],
                    data_loader=batches,
                    forward_step=_tensorrt_modelopt_torch_forward_step,
                    loss_func=loss_func if method == "gradient" else None,
                    num_calib_steps=calib_steps,
                    num_score_steps=score_steps,
                    method=method,
                    verbose=_env_bool("HDRTVNET_TRT_INT8_MODELOPT_TORCH_VERBOSE", False),
                )
        except Exception as exc:
            if method == "kl_div" or fallback_model is None:
                raise
            print(
                "TensorRT INT8 ModelOpt Torch gradient search failed; "
                f"retrying kl_div search ({exc})"
            )
            _disable_tensorrt_modelopt_torch_inplace_activations(fallback_model)
            with _tensorrt_modelopt_torch_no_inplace_functionals():
                batches = _make_tensorrt_modelopt_torch_batches(
                    model=fallback_model,
                    width=width,
                    height=height,
                    dtype=dtype,
                    device=device,
                    steps=max(calib_steps, score_steps),
                    include_targets=False,
                )
                quantized, _search_state = mtq.auto_quantize(
                    fallback_model,
                    constraints=constraints,
                    quantization_formats=[copy.deepcopy(quant_config)],
                    data_loader=batches,
                    forward_step=_tensorrt_modelopt_torch_forward_step,
                    num_calib_steps=calib_steps,
                    num_score_steps=score_steps,
                    method="kl_div",
                    verbose=_env_bool("HDRTVNET_TRT_INT8_MODELOPT_TORCH_VERBOSE", False),
                )
        print(
            "TensorRT INT8 ModelOpt Torch auto_quantize: "
            f"effective_bits={constraints['effective_bits']:g}, method={method}"
        )

    _apply_tensorrt_modelopt_torch_quantizer_filters(quantized, precision)
    if full_precision:
        _enable_tensorrt_modelopt_torch_quantizers(
            quantized,
            kinds=("input_quantizer", "weight_quantizer", "output_quantizer"),
            label="Full",
        )
        _enable_tensorrt_modelopt_torch_output_quantizers(quantized)
        _calibrate_tensorrt_modelopt_torch_enabled_activation_quantizers(
            quantized,
            forward_loop,
            kinds=("input_quantizer", "output_quantizer"),
            label="full activation",
        )
    elif _env_bool("HDRTVNET_TRT_INT8_MODELOPT_TORCH_OUTPUTS", False):
        _enable_tensorrt_modelopt_torch_filtered_output_quantizers(quantized)
        _calibrate_tensorrt_modelopt_torch_enabled_output_quantizers(
            quantized,
            forward_loop,
        )
    enabled, total = _count_tensorrt_modelopt_torch_quantizers(quantized)
    if total:
        suffix = (
            "all quantizers required"
            if full_precision
            else "all, output quantizers may stay disabled"
        )
        print(f"TensorRT INT8 ModelOpt Torch quantizers ({suffix}): {enabled}/{total} enabled")
        _print_tensorrt_modelopt_torch_quantizer_summary(quantized)
    if full_precision:
        _enforce_tensorrt_modelopt_torch_full_quantizers(quantized)
    try:
        quantized.eval()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return quantized


def _export_tensorrt_modelopt_torch_onnx_from_model(
    *,
    model: nn.Module,
    onnx_path: str,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
    flat_model: bool,
    single_input_graph: bool = False,
) -> str:
    _remove_onnx_artifacts(onnx_path)
    export_model = getattr(model, "_orig_mod", model).eval()
    _set_tensorrt_modelopt_torch_export_dtype(export_model, dtype)
    _patch_adaptive_avgpool_for_trt(export_model)
    h, w = int(height), int(width)
    cond_h, cond_w = max(1, h // 4), max(1, w // 4)
    tensor = torch.zeros((1, 3, h, w), dtype=dtype, device=device)
    cond = torch.zeros((1, 3, cond_h, cond_w), dtype=dtype, device=device)
    wrapper = _ONNXExportWrapper(
        export_model,
        flat_model=flat_model,
        allow_missing_cond=single_input_graph,
    ).eval()
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    if single_input_graph:
        print("TensorRT ModelOpt Torch ONNX export: single input enabled")
    print(f"TensorRT ModelOpt Torch ONNX export: {onnx_path}")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    try:
        from modelopt.torch.quantization.export_onnx import (
            configure_linear_module_onnx_quantizers,
        )
        from modelopt.torch.quantization.utils import export_torch_mode
        import modelopt.torch.quantization.tensor_quant as modelopt_tensor_quant
    except Exception:
        configure_linear_module_onnx_quantizers = None
        export_torch_mode = None
        modelopt_tensor_quant = None

    def _export_passthrough_quantize_op(
        inputs: torch.Tensor,
        amax: torch.Tensor,
        num_bits: int = 8,
        exponent_bits: int = 0,
        unsigned: bool = False,
        narrow_range: bool = True,
    ) -> torch.Tensor:
        return inputs

    original_quantize_op = None
    if modelopt_tensor_quant is not None:
        original_quantize_op = getattr(modelopt_tensor_quant, "quantize_op", None)
        modelopt_tensor_quant.quantize_op = _export_passthrough_quantize_op
    export_cm = (
        export_torch_mode()
        if (
            export_torch_mode is not None
            and _env_bool("HDRTVNET_TRT_INT8_MODELOPT_TORCH_EXPORT_MODE", False)
        )
        else contextlib.nullcontext()
    )
    quantizer_cm = (
        configure_linear_module_onnx_quantizers(export_model)
        if configure_linear_module_onnx_quantizers is not None
        else contextlib.nullcontext()
    )
    with torch.inference_mode():
        try:
            with export_cm, quantizer_cm:
                export_args = (tensor,) if single_input_graph else (tensor, cond)
                input_names = ["input"] if single_input_graph else ["input", "cond"]
                torch.onnx.export(
                    wrapper,
                    export_args,
                    onnx_path,
                    verbose=False,
                    input_names=input_names,
                    output_names=["output"],
                    opset_version=19,
                    do_constant_folding=True,
                    dynamo=False,
                )
        finally:
            if modelopt_tensor_quant is not None and original_quantize_op is not None:
                modelopt_tensor_quant.quantize_op = original_quantize_op
    _patch_tensorrt_qdq_zero_points(onnx_path)
    if dtype == torch.float16:
        _patch_tensorrt_fp16_constants(onnx_path)
    return onnx_path


def _tensorrt_modelopt_calibration_shapes(
    onnx_path: str,
    width: int,
    height: int,
) -> str:
    input_names: list[str] = []
    try:
        import onnx

        model = onnx.load(onnx_path, load_external_data=False)
        input_names = [str(inp.name) for inp in model.graph.input]
    except Exception:
        input_names = ["input", "cond"]

    h, w = int(height), int(width)
    cond_h, cond_w = max(1, h // 4), max(1, w // 4)
    shapes: list[str] = []
    for name in input_names:
        if name == "input":
            dims = (1, 3, h, w)
        elif name == "cond":
            dims = (1, 3, cond_h, cond_w)
        else:
            dims = (1,)
        shapes.append(f"{name}:{'x'.join(str(int(v)) for v in dims)}")
    return ",".join(shapes)


def _patch_tensorrt_modelopt_autotune_region_api() -> None:
    """Patch ModelOpt autotune Region API drift in the installed package."""
    try:
        from modelopt.onnx.quantization.autotune.common import Region
    except Exception:
        return
    if hasattr(Region, "get_all_nodes_recursive"):
        return

    def _get_all_nodes_recursive(self):
        return sorted(self.get_region_nodes_and_descendants())

    Region.get_all_nodes_recursive = _get_all_nodes_recursive


def _apply_tensorrt_modelopt_int8_quantization(
    onnx_path: str,
    *,
    width: int,
    height: int,
    nodes_to_quantize: list[str] | None = None,
) -> str:
    if not _tensorrt_int8_modelopt_enabled():
        return onnx_path

    root, ext = os.path.splitext(onnx_path)
    quantized_path = f"{root}_modelopt_int8{ext or '.onnx'}"
    prequantized = _tensorrt_int8_modelopt_prequantized_onnx(onnx_path)
    if prequantized:
        prequantized_path = os.path.abspath(os.path.expanduser(prequantized))
        if not os.path.isfile(prequantized_path):
            message = f"TensorRT INT8 ModelOpt prequantized ONNX missing: {prequantized_path}"
            print(message)
            raise RuntimeError(message)
        _remove_onnx_artifacts(quantized_path)
        try:
            import onnx

            model = onnx.load(prequantized_path, load_external_data=True)
            onnx.save_model(model, quantized_path, save_as_external_data=False)
        except Exception:
            shutil.copy2(prequantized_path, quantized_path)
            data_path = f"{prequantized_path}.data"
            if os.path.isfile(data_path):
                shutil.copy2(data_path, f"{quantized_path}.data")
        _patch_tensorrt_qdq_zero_points(quantized_path)
        if _env_bool("HDRTVNET_TRT_INT8_MODELOPT_FP32_SCALES", False):
            _patch_tensorrt_fp16_qdq_scales(quantized_path)
        print(f"TensorRT INT8 ModelOpt prequantized ONNX: {prequantized_path}")
        print(f"TensorRT INT8 ModelOpt ONNX: {quantized_path}")
        return quantized_path

    try:
        from modelopt.onnx.quantization import quantize as modelopt_quantize
    except Exception as exc:
        message = f"TensorRT INT8 ModelOpt quantization unavailable: {exc}"
        print(message)
        raise RuntimeError(message) from exc

    _remove_onnx_artifacts(quantized_path)
    calibration_shapes = _tensorrt_modelopt_calibration_shapes(
        onnx_path,
        width,
        height,
    )
    calibration_method = _tensorrt_int8_modelopt_calibration_method()
    ops = _tensorrt_int8_modelopt_ops()
    autotune_enabled = _tensorrt_int8_modelopt_autotune_enabled()
    autotune_filter = _tensorrt_int8_modelopt_autotune_filter()
    autotune_output_dir = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_DIR", "")
    ).strip()
    if not autotune_output_dir:
        autotune_output_dir = os.path.join(
            os.path.dirname(quantized_path),
            "modelopt_autotune",
            pathlib.Path(quantized_path).stem,
        )
    log_level = str(
        os.environ.get("HDRTVNET_TRT_INT8_MODELOPT_LOG", "WARNING")
    ).strip().upper() or "WARNING"

    ops_label = "all supported ops" if ops is None else ",".join(ops)
    node_label = (
        "all matching nodes"
        if not nodes_to_quantize
        else f"{len(nodes_to_quantize)} checkpoint-selected node(s)"
    )
    print(
        "TensorRT INT8 ModelOpt: inserting explicit INT8 Q/DQ "
        f"(calibration={calibration_method}, ops={ops_label}, "
        f"nodes={node_label}, shapes={calibration_shapes})"
        )
    if autotune_enabled:
        _patch_tensorrt_modelopt_autotune_region_api()
        print(
            "TensorRT INT8 ModelOpt autotune: "
            f"schemes={_tensorrt_int8_modelopt_autotune_int('HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_SCHEMES', 12, 1)}, "
            f"warmup={_tensorrt_int8_modelopt_autotune_int('HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_WARMUP', 8, 0)}, "
            f"runs={_tensorrt_int8_modelopt_autotune_int('HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_RUNS', 16, 1)}, "
            f"filter={autotune_filter or 'none'}"
        )
    try:
        modelopt_quantize(
            onnx_path,
            quantize_mode="int8",
            calibration_method=calibration_method,
            calibration_shapes=calibration_shapes,
            calibration_eps=["cpu"],
            output_path=quantized_path,
            high_precision_dtype="fp16",
            op_types_to_quantize=ops,
            nodes_to_quantize=nodes_to_quantize,
            use_external_data_format=False,
            keep_intermediate_files=_env_bool(
                "HDRTVNET_TRT_INT8_MODELOPT_KEEP_INTERMEDIATE",
                False,
            ),
            log_level=log_level,
            use_zero_point=False,
            opset=19,
            autotune=autotune_enabled,
            autotune_output_dir=autotune_output_dir,
            autotune_num_schemes_per_region=_tensorrt_int8_modelopt_autotune_int(
                "HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_SCHEMES",
                12,
                1,
            ),
            autotune_node_filter_list=autotune_filter,
            autotune_verbose=_env_bool(
                "HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_VERBOSE",
                False,
            ),
            autotune_use_trtexec=False,
            autotune_timing_cache=_tensorrt_timing_cache_path(),
            autotune_warmup_runs=_tensorrt_int8_modelopt_autotune_int(
                "HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_WARMUP",
                8,
                0,
            ),
            autotune_timing_runs=_tensorrt_int8_modelopt_autotune_int(
                "HDRTVNET_TRT_INT8_MODELOPT_AUTOTUNE_RUNS",
                16,
                1,
            ),
        )
    except Exception as exc:
        message = f"TensorRT INT8 ModelOpt quantization failed: {exc}"
        print(message)
        raise RuntimeError(message) from exc

    if not os.path.isfile(quantized_path):
        message = "TensorRT INT8 ModelOpt quantization produced no ONNX."
        print(message)
        raise RuntimeError(message)
    _patch_tensorrt_qdq_zero_points(quantized_path)
    if _env_bool("HDRTVNET_TRT_INT8_MODELOPT_FP32_SCALES", False):
        _patch_tensorrt_fp16_qdq_scales(quantized_path)
    print(f"TensorRT INT8 ModelOpt ONNX: {quantized_path}")
    return quantized_path


def _resolve_tensorrt_calibration_frames(value=None) -> int:
    if value is None:
        value = os.environ.get("HDRTVNET_TRT_CALIBRATION_FRAMES", "64")
    try:
        frames = int(value)
    except Exception:
        frames = 64
    # 0 means "all available" for dataset/video calibration inputs.
    return max(0, frames)


def _tensorrt_calibration_cache_path(engine_path: str, override: str | None = None) -> str:
    path = str(override or "").strip()
    if path:
        return os.path.abspath(os.path.expanduser(path))
    return os.path.splitext(str(engine_path))[0] + ".calib"


def _tensorrt_calibration_cond(tensor: torch.Tensor) -> torch.Tensor:
    try:
        cond = F.interpolate(
            tensor,
            scale_factor=0.25,
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
            antialias=True,
        )
    except TypeError:
        cond = F.interpolate(
            tensor,
            scale_factor=0.25,
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
    return cond.contiguous()


def _tensorrt_frame_to_calibration_batch(
    frame_bgr,
    *,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    import cv2

    if frame_bgr.shape[1] != int(width) or frame_bgr.shape[0] != int(height):
        frame_bgr = cv2.resize(
            frame_bgr,
            (int(width), int(height)),
            interpolation=cv2.INTER_AREA,
        )
    frame_bgr = np.ascontiguousarray(frame_bgr)
    raw = torch.from_numpy(frame_bgr).to(device=device)
    tensor = (
        raw.flip(2)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=dtype)
        .mul(1.0 / 255.0)
        .contiguous()
    )
    return {
        "input": tensor,
        "cond": _tensorrt_calibration_cond(tensor),
    }


def _tensorrt_calibration_image_paths(dataset_path: str) -> list[pathlib.Path]:
    root = pathlib.Path(str(dataset_path)).expanduser()
    if root.is_file() and root.suffix.lower() in {".txt", ".lst", ".csv"}:
        base_dir = root.parent
        paths = []
        for raw_line in root.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip().strip('"')
            if not line or line.startswith("#"):
                continue
            candidate = pathlib.Path(line).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            if candidate.is_file():
                paths.append(candidate)
        return paths
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"TensorRT calibration dataset not found: {dataset_path}")
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def _tensorrt_calibration_image_score(path: pathlib.Path) -> tuple[float, int] | None:
    try:
        resolved = path.resolve()
        stat = resolved.stat()
        cache_key = (
            os.path.normcase(str(resolved)),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
    except Exception:
        return None
    cached = _TENSORRT_CALIBRATION_IMAGE_SCORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import cv2

        frame = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        max_dim = max(int(h), int(w))
        if max_dim > 320:
            scale = 320.0 / float(max_dim)
            frame = cv2.resize(
                frame,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32) * (1.0 / 255.0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) * (1.0 / 255.0)

        mean_y = float(np.mean(y))
        std_y = float(np.std(y))
        p01, p99 = np.percentile(y, [1.0, 99.0])
        dynamic = float(max(0.0, p99 - p01))
        highlight = float(np.mean(y >= 0.90))
        shadow = float(np.mean(y <= 0.08))
        sat_mean = float(np.mean(sat))

        hist = np.histogram(y, bins=64, range=(0.0, 1.0))[0].astype(np.float32)
        hist_sum = float(hist.sum())
        if hist_sum > 0.0:
            prob = hist / hist_sum
            prob = prob[prob > 0.0]
            entropy = float(-(prob * np.log2(prob)).sum() / math.log2(64.0))
        else:
            entropy = 0.0

        exposure_balance = 1.0 - min(1.0, abs(mean_y - 0.50) * 2.0)
        highlight_score = min(highlight / 0.08, 1.0)
        shadow_score = min(shadow / 0.08, 1.0)
        lap = cv2.Laplacian(y, cv2.CV_32F)
        edge_score = min(float(np.var(lap)) / 0.015, 1.0)

        score = (
            dynamic * 3.0
            + std_y * 2.0
            + entropy * 2.0
            + sat_mean * 1.0
            + exposure_balance * 0.75
            + highlight_score * 0.6
            + shadow_score * 0.6
            + edge_score * 0.4
        )
        bucket = int(np.clip(math.floor(mean_y * 8.0), 0, 7))
    except Exception:
        return None

    result = (float(score), int(bucket))
    _TENSORRT_CALIBRATION_IMAGE_SCORE_CACHE[cache_key] = result
    return result


def _select_tensorrt_calibration_image_paths(
    paths: list[pathlib.Path],
    count: int,
) -> tuple[list[pathlib.Path], str]:
    if count <= 0 or len(paths) <= count:
        return list(paths), "all"

    scored: list[tuple[float, int, int, pathlib.Path]] = []
    for index, path in enumerate(paths):
        score_bucket = _tensorrt_calibration_image_score(path)
        if score_bucket is None:
            continue
        score, bucket = score_bucket
        scored.append((float(score), int(bucket), int(index), path))

    if len(scored) < count:
        indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
        return [paths[int(i)] for i in indices], "uniform-fallback"

    ranked = sorted(scored, key=lambda item: (-item[0], item[2]))
    buckets: dict[int, list[tuple[float, int, int, pathlib.Path]]] = {
        bucket: [] for bucket in range(8)
    }
    for item in ranked:
        buckets[item[1]].append(item)

    selected: dict[int, tuple[float, int, int, pathlib.Path]] = {}
    base_quota = count // 8
    remainder = count % 8
    bucket_order = sorted(
        range(8),
        key=lambda bucket: (
            -len(buckets[bucket]),
            -buckets[bucket][0][0] if buckets[bucket] else float("-inf"),
            bucket,
        ),
    )
    quotas = {bucket: base_quota for bucket in range(8)}
    for bucket in bucket_order[:remainder]:
        quotas[bucket] += 1

    for bucket in range(8):
        for item in buckets[bucket][:quotas[bucket]]:
            selected[item[2]] = item

    if len(selected) < count:
        for item in ranked:
            selected.setdefault(item[2], item)
            if len(selected) >= count:
                break

    selected_items = sorted(selected.values(), key=lambda item: item[2])
    return [item[3] for item in selected_items[:count]], "content-ranked"


class _TensorRTCalibrationSource:
    def next_batch(self) -> dict[str, torch.Tensor] | None:
        return None

    def close(self) -> None:
        return None


class _TensorRTListCalibrationSource(_TensorRTCalibrationSource):
    def __init__(self, batches: list[dict[str, torch.Tensor]]):
        self._batches = list(batches)
        self._index = 0

    def next_batch(self) -> dict[str, torch.Tensor] | None:
        if self._index >= len(self._batches):
            return None
        batch = self._batches[self._index]
        self._index += 1
        return batch


class _TensorRTDatasetCalibrationSource(_TensorRTCalibrationSource):
    def __init__(
        self,
        *,
        dataset_path: str,
        width: int,
        height: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_count: int,
    ):
        paths = _tensorrt_calibration_image_paths(dataset_path)
        if not paths:
            raise RuntimeError(f"No calibration images found under: {dataset_path}")
        requested = _resolve_tensorrt_calibration_frames(frame_count)
        count = len(paths) if requested <= 0 else min(len(paths), requested)
        self._paths, selection_mode = _select_tensorrt_calibration_image_paths(
            paths,
            count,
        )
        self._dataset_path = dataset_path
        self._width = int(width)
        self._height = int(height)
        self._dtype = dtype
        self._device = device
        self._index = 0
        print(
            "TensorRT native INT8 calibration: "
            f"streaming {len(self._paths)} image(s) from {dataset_path} "
            f"({selection_mode} selection)"
        )

    def next_batch(self) -> dict[str, torch.Tensor] | None:
        import cv2

        with torch.inference_mode():
            while self._index < len(self._paths):
                path = self._paths[self._index]
                self._index += 1
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                return _tensorrt_frame_to_calibration_batch(
                    frame,
                    width=self._width,
                    height=self._height,
                    dtype=self._dtype,
                    device=self._device,
                )
        return None


class _TensorRTVideoCalibrationSource(_TensorRTCalibrationSource):
    def __init__(
        self,
        *,
        video_path: str,
        width: int,
        height: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_count: int,
    ):
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open TensorRT calibration video: {video_path}")
        self._video_path = video_path
        self._width = int(width)
        self._height = int(height)
        self._dtype = dtype
        self._device = device
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        requested = _resolve_tensorrt_calibration_frames(frame_count)
        if self._total > 1 and requested > 0:
            count = min(self._total, requested)
            self._positions = [
                int(v)
                for v in np.linspace(0, max(0, self._total - 1), count, dtype=np.int64)
            ]
            label_count = len(self._positions)
        elif self._total <= 1 and requested > 0:
            self._positions = list(range(requested))
            label_count = len(self._positions)
        else:
            self._positions = None
            label_count = self._total if self._total > 0 else "all available"
        self._index = 0
        print(
            "TensorRT native INT8 calibration: "
            f"streaming {label_count} frame(s) from {video_path}"
        )

    def close(self) -> None:
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None

    def next_batch(self) -> dict[str, torch.Tensor] | None:
        cap = getattr(self, "_cap", None)
        if cap is None:
            return None
        with torch.inference_mode():
            while True:
                if self._positions is None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self.close()
                        return None
                else:
                    if self._index >= len(self._positions):
                        self.close()
                        return None
                    pos = int(self._positions[self._index])
                    self._index += 1
                    if self._total > 1:
                        cap.set(self._cv2.CAP_PROP_POS_FRAMES, pos)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                return _tensorrt_frame_to_calibration_batch(
                    frame,
                    width=self._width,
                    height=self._height,
                    dtype=self._dtype,
                    device=self._device,
                )


class _TensorRTSyntheticCalibrationSource(_TensorRTCalibrationSource):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_count: int,
    ):
        self._count = max(1, _resolve_tensorrt_calibration_frames(frame_count))
        self._index = 0
        h, w = int(height), int(width)
        with torch.inference_mode():
            self._ramp_x = torch.linspace(0.0, 1.0, w, dtype=dtype, device=device)
            self._ramp_x = self._ramp_x.view(1, 1, 1, w).expand(1, 3, h, w)
            self._ramp_y = torch.linspace(0.0, 1.0, h, dtype=dtype, device=device)
            self._ramp_y = self._ramp_y.view(1, 1, h, 1).expand(1, 3, h, w)
        print(
            "TensorRT native INT8 calibration: "
            f"using {self._count} synthetic frame(s)"
        )

    def next_batch(self) -> dict[str, torch.Tensor] | None:
        if self._index >= self._count:
            return None
        idx = self._index
        self._index += 1
        with torch.inference_mode():
            mix = float(idx % 8) / 7.0 if self._count > 1 else 0.5
            tensor = (self._ramp_x * mix + self._ramp_y * (1.0 - mix)).contiguous()
            return {
                "input": tensor,
                "cond": _tensorrt_calibration_cond(tensor),
            }


def _make_tensorrt_dataset_calibration_source(
    *,
    dataset_path: str,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
    frame_count: int,
) -> _TensorRTCalibrationSource:
    return _TensorRTDatasetCalibrationSource(
        dataset_path=dataset_path,
        width=width,
        height=height,
        dtype=dtype,
        device=device,
        frame_count=frame_count,
    )


def _make_tensorrt_video_calibration_source(
    *,
    video_path: str,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
    frame_count: int,
) -> _TensorRTCalibrationSource:
    return _TensorRTVideoCalibrationSource(
        video_path=video_path,
        width=width,
        height=height,
        dtype=dtype,
        device=device,
        frame_count=frame_count,
    )


def _make_synthetic_tensorrt_calibration_source(
    *,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
    frame_count: int,
) -> _TensorRTCalibrationSource:
    return _TensorRTSyntheticCalibrationSource(
        width=width,
        height=height,
        dtype=dtype,
        device=device,
        frame_count=frame_count,
    )


def _make_tensorrt_int8_calibrator(
    trt,
    *,
    cache_path: str,
    calibration_source: _TensorRTCalibrationSource | None,
):
    algorithm = _tensorrt_calibrator_algorithm()
    if algorithm == "minmax":
        base_cls = getattr(trt, "IInt8MinMaxCalibrator", None)
        if base_cls is None:
            base_cls = getattr(trt, "IInt8EntropyCalibrator2", None)
            algorithm = "entropy"
    else:
        base_cls = getattr(trt, "IInt8EntropyCalibrator2", None)
        if base_cls is None:
            base_cls = getattr(trt, "IInt8MinMaxCalibrator", None)
            algorithm = "minmax"
    if base_cls is None:
        raise RuntimeError("TensorRT INT8 calibrator API is unavailable.")
    print(f"TensorRT native INT8 calibrator: {algorithm}")

    class _Calibrator(base_cls):
        def __init__(
            self,
            calibration_cache: str,
            source: _TensorRTCalibrationSource | None,
        ):
            base_cls.__init__(self)
            self._cache_path = calibration_cache
            self._source = source
            self._active_batch = None

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self._source is None:
                return None
            batch = self._source.next_batch()
            if batch is None:
                self._source.close()
                return None
            self._active_batch = batch
            ordered = []
            fallback = ("input", "cond")
            for idx, name in enumerate(names):
                key = str(name)
                tensor = batch.get(key)
                if tensor is None and idx < len(fallback):
                    tensor = batch.get(fallback[idx])
                if tensor is None:
                    raise RuntimeError(
                        f"TensorRT calibration input not found: {key}"
                    )
                ordered.append(int(tensor.data_ptr()))
            return ordered

        def __del__(self):
            try:
                if self._source is not None:
                    self._source.close()
            except Exception:
                pass

        def read_calibration_cache(self):
            try:
                if self._cache_path and os.path.isfile(self._cache_path):
                    print(
                        "TensorRT native INT8 calibration cache loaded: "
                        f"{self._cache_path}"
                    )
                    return pathlib.Path(self._cache_path).read_bytes()
            except Exception as exc:
                print(f"TensorRT calibration cache read skipped: {exc}")
            return None

        def write_calibration_cache(self, cache):
            if not self._cache_path:
                return
            try:
                path = pathlib.Path(self._cache_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes(cache))
                print(f"TensorRT native INT8 calibration cache saved: {path}")
            except Exception as exc:
                print(f"TensorRT calibration cache write skipped: {exc}")

    return _Calibrator(cache_path, calibration_source)


class HDRTVNetTensorRT(HDRTVNetTorch):
    """TensorRT runtime with PyTorch preprocessing/postprocessing parity.

    Engines are built lazily for the selected model/resolution/mode and then
    reused from disk. INT8 checkpoints can be exported either as native ONNX
    layers with TensorRT PTQ calibration or as explicit Q/DQ graphs.
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
        predequantize=False,
        qdq_fusion: str = "native",
        keep_onnx: bool = False,
        calibration_dataset: str | None = None,
        calibration_video: str | None = None,
        calibration_frames: int | None = None,
        calibration_cache: str | None = None,
    ):
        if not _IS_NVIDIA:
            raise RuntimeError("TensorRT backend is only enabled for NVIDIA CUDA devices.")

        model_path = tensorrt_source_checkpoint_path(model_path)
        precision_text = str(precision or "").strip().lower()
        self._trt_direct_fp_int8 = (
            precision_text.startswith("int8")
            and not _is_int8_checkpoint_path(model_path)
        )
        self._engine_width = int(engine_width)
        self._engine_height = int(engine_height)
        self._trt_base_mode_name = str(mode_name or precision or "mode")
        self._trt_qdq_fusion = _resolve_tensorrt_qdq_fusion(precision, qdq_fusion)
        self._trt_predequantize_int8 = _resolve_tensorrt_predequantize(
            precision,
            predequantize,
        )
        self._trt_modelopt_int8 = (
            precision_text.startswith("int8")
            and not self._trt_predequantize_int8
            and _tensorrt_int8_modelopt_enabled()
        )
        self._trt_qat_checkpoint_composition = (
            self._trt_modelopt_int8
            and _tensorrt_int8_qat_checkpoint_composition_enabled(
                self._trt_base_mode_name
            )
        )
        self._trt_modelopt_torch_int8 = (
            self._trt_modelopt_int8
            and _tensorrt_int8_modelopt_torch_effective_enabled(
                self._trt_base_mode_name
            )
        )
        self._trt_int8_agcm_only = (
            precision_text.startswith("int8")
            and not self._trt_modelopt_int8
            and _tensorrt_int8_agcm_only_enabled(self._trt_base_mode_name)
        )
        self._trt_int8_zero_condition = (
            precision_text.startswith("int8")
            and not self._trt_predequantize_int8
            and not self._trt_modelopt_int8
            and _tensorrt_int8_zero_condition_enabled(self._trt_base_mode_name)
        )
        self._trt_int8_single_input = (
            precision_text.startswith("int8")
            and not self._trt_predequantize_int8
            and not self._trt_modelopt_int8
            and _tensorrt_int8_single_input_enabled(self._trt_base_mode_name)
        )
        self._trt_builder_precision = (
            "fp16" if self._trt_predequantize_int8 else precision_text
        )
        self._engine_mode_name = tensorrt_mode_name(
            precision,
            self._trt_base_mode_name,
            predequantize=self._trt_predequantize_int8,
            qdq_fusion=self._trt_qdq_fusion,
        )
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
        self._trt_keep_onnx = bool(keep_onnx)
        (
            self._trt_calibration_dataset,
            self._trt_calibration_video,
            self._trt_calibration_cache,
            self._trt_used_prebuilt_calibration_cache,
        ) = _resolve_tensorrt_calibration_sources(
            model_path=model_path,
            width=self._engine_width,
            height=self._engine_height,
            precision=precision,
            mode_name=self._trt_base_mode_name,
            use_hg=use_hg,
            predequantize=self._trt_predequantize_int8,
            qdq_fusion=self._trt_qdq_fusion,
            calibration_dataset=calibration_dataset,
            calibration_video=calibration_video,
            calibration_cache=calibration_cache,
        )
        self._trt_calibration_frames = _resolve_tensorrt_calibration_frames(
            calibration_frames
        )
        self._trt_int8_calibrator = None
        self._trt_context = None
        self._trt_input_names = []
        self._trt_output_names = []
        self._trt_output = None
        self._trt_output_shape = None
        self._trt_shape_cache_key = None
        self._trt_address_cache_key = None
        self._trt_legacy_shape_cache_key = None
        self._trt_use_dedicated_stream = _env_bool(
            "HDRTVNET_TRT_DEDICATED_STREAM",
            True,
        )
        self._trt_stream = None
        super().__init__(
            model_path,
            device=device,
            precision=(
                "fp16"
                if self._trt_direct_fp_int8
                else precision
            ),
            compile_model=False,
            force_compile=False,
            compile_mode="default",
            use_cuda_graphs=False,
            force_channels_last=False,
            predequantize=(
                True
                if (self._trt_predequantize_int8 or self._trt_modelopt_int8)
                else False
            ),
            hg_weights=hg_weights,
            use_hg=use_hg,
            warmup_passes=0,
        )
        if self._trt_direct_fp_int8:
            self.precision = precision_text
        if self._trt_int8_zero_condition:
            self._fast_zero_condition = True
            print("TensorRT INT8 speed mode: zero condition tensor enabled")
        if self._trt_int8_single_input:
            print("TensorRT INT8 speed mode: single input enabled")
        if self._trt_int8_agcm_only:
            print("TensorRT INT8 speed mode: AGCM-only output enabled")

        # TensorRT bindings use dense NCHW buffers; keep preprocessing and
        # postprocessing identical, but make the staging tensors contiguous.
        self._use_channels_last = False
        if self._trt_use_dedicated_stream:
            self._trt_stream = torch.cuda.Stream()
        if self._trt_predequantize_int8:
            if str(precision or "").startswith("int8-full"):
                print(
                    "TensorRT full INT8 safety: using an AGCM_LE FP16 builder "
                    "safe engine for this full INT8 preset"
                )
            else:
                print(
                    "TensorRT mixed balance: exporting an AGCM_LE FP16 builder "
                    "engine for the selected INT8 mixed preset"
                )
        elif self._trt_modelopt_torch_int8:
            print(
                "TensorRT INT8 ModelOpt Torch: quantizing the PyTorch model "
                "before ONNX export, then building a native TensorRT engine"
            )
        elif self._trt_modelopt_int8:
            if getattr(self, "_trt_qat_checkpoint_composition", False):
                print(
                    "TensorRT INT8 QAT composition: honoring checkpoint "
                    "W8A8/FP16 layer metadata, then inserting explicit INT8 "
                    "Q/DQ for TensorRT"
                )
            else:
                print(
                    "TensorRT INT8 ModelOpt: exporting selected weights as FP16 "
                    "then inserting explicit INT8 Q/DQ for TensorRT"
                )
        elif self._trt_direct_fp_int8:
            print(
                "TensorRT native INT8: exporting the FP16 AGCM_LE model and "
                "letting TensorRT choose native INT8 tactics"
            )
        if getattr(self, "_trt_used_prebuilt_calibration_cache", False):
            print(
                "TensorRT prebuilt INT8 calibration cache: "
                f"{self._trt_calibration_cache}"
            )

        self._trt_engine_metadata = self._expected_tensorrt_engine_metadata()
        engine_validation_error = _tensorrt_engine_validation_error(
            self.engine_path,
            self._trt_engine_metadata,
        )
        engine_rebuilt = False
        if engine_validation_error:
            if os.path.isfile(self.engine_path):
                print(
                    "TensorRT engine cache invalid: "
                    f"{engine_validation_error} ({self.engine_path})"
                )
            else:
                print(f"TensorRT engine cache miss: {self.engine_path}")
            self._build_engine_from_loaded_model()
            engine_rebuilt = True
        else:
            print(f"TensorRT engine cache hit: {self.engine_path}")
            if not self._trt_keep_onnx:
                cleanup_tensorrt_onnx_after_engine(self.onnx_path, self.engine_path)

        try:
            self._load_engine()
        except Exception as exc:
            if engine_rebuilt:
                raise
            print(f"TensorRT cached engine load failed; rebuilding: {exc}")
            self._trt_runtime = None
            self._trt_engine = None
            self._trt_context = None
            self._trt_input_names = []
            self._trt_output_names = []
            self._build_engine_from_loaded_model()
            self._load_engine()

        # Runtime must not retain the PyTorch model after the engine exists.
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._compiled = False
        print(f"TensorRT engine : {self.engine_path}")

    def _expected_tensorrt_engine_metadata(self, trt_module=None) -> dict[str, object]:
        return _tensorrt_expected_engine_metadata(
            model_path=self.model_path,
            width=self._engine_width,
            height=self._engine_height,
            precision=self.precision,
            mode_name=str(
                getattr(
                    self,
                    "_trt_base_mode_name",
                    self._engine_mode_name,
                )
            ),
            engine_mode_name=self._engine_mode_name,
            builder_precision=self._trt_builder_precision,
            use_hg=self._use_hg,
            predequantize_int8=self._trt_predequantize_int8,
            qdq_fusion=self._trt_qdq_fusion,
            hg_weights=self._hg_weights,
            calibration_dataset=getattr(self, "_trt_calibration_dataset", None),
            calibration_video=getattr(self, "_trt_calibration_video", None),
            calibration_frames=getattr(self, "_trt_calibration_frames", None),
            calibration_cache=getattr(self, "_trt_calibration_cache", None),
            trt_module=trt_module,
        )

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

        self._configure_assume_aligned_shapes(
            self._engine_width,
            self._engine_height,
        )
        single_input_graph = _tensorrt_condition_free_arch_enabled_for_model(self.model)
        if getattr(self, "_trt_modelopt_torch_int8", False):
            self.model = _apply_tensorrt_modelopt_torch_int8_quantization(
                model=self.model,
                width=self._engine_width,
                height=self._engine_height,
                dtype=self._dtype,
                device=self.device,
                precision=self.precision,
            )
            onnx_path = _export_tensorrt_modelopt_torch_onnx_from_model(
                model=self.model,
                onnx_path=self.onnx_path,
                width=self._engine_width,
                height=self._engine_height,
                dtype=self._dtype,
                device=self.device,
                flat_model=getattr(self, "_is_flat_model", False),
                single_input_graph=single_input_graph,
            )
        else:
            export_precision = (
                "fp16"
                if getattr(self, "_trt_modelopt_int8", False)
                else self._trt_builder_precision
            )
            onnx_path = _export_tensorrt_onnx_from_model(
                model=self.model,
                onnx_path=self.onnx_path,
                width=self._engine_width,
                height=self._engine_height,
                dtype=self._dtype,
                device=self.device,
                precision=export_precision,
                flat_model=getattr(self, "_is_flat_model", False),
                qdq_fusion=self._trt_qdq_fusion,
                int8_zero_condition=getattr(self, "_trt_int8_zero_condition", False),
                int8_single_input=getattr(self, "_trt_int8_single_input", False),
                int8_agcm_only=getattr(self, "_trt_int8_agcm_only", False),
                single_input_graph=single_input_graph,
            )
        build_onnx_path = onnx_path
        if (
            getattr(self, "_trt_modelopt_int8", False)
            and not getattr(self, "_trt_modelopt_torch_int8", False)
        ):
            force_checkpoint_policy = bool(
                getattr(self, "_trt_qat_checkpoint_composition", False)
            )
            nodes_to_quantize = None
            if not force_checkpoint_policy:
                nodes_to_quantize = (
                    None
                    if _tensorrt_int8_modelopt_autotune_enabled()
                    else _tensorrt_int8_modelopt_node_override()
                )
            if nodes_to_quantize:
                print(
                    "TensorRT INT8 ModelOpt node override: "
                    f"{len(nodes_to_quantize)} node(s)"
                )
            elif force_checkpoint_policy or (
                _tensorrt_int8_modelopt_layer_policy() == "checkpoint"
            ):
                checkpoint_policy = _tensorrt_native_int8_onnx_policy(
                    onnx_path,
                    quant_type=getattr(self, "_int8_quant_type", None),
                    w8a8_layers=getattr(self, "_int8_w8a8_layers", None),
                )
                nodes_to_quantize = sorted(
                    name
                    for name, precision in checkpoint_policy.items()
                    if precision == "int8"
                )
                if not nodes_to_quantize:
                    nodes_to_quantize = None
            build_onnx_path = _apply_tensorrt_modelopt_int8_quantization(
                onnx_path,
                width=self._engine_width,
                height=self._engine_height,
                nodes_to_quantize=nodes_to_quantize,
            )
        self._build_engine_from_onnx(build_onnx_path, trt)
        if self._trt_keep_onnx:
            print(f"TensorRT ONNX kept for inspection: {onnx_path}")
            if build_onnx_path != onnx_path:
                print(f"TensorRT build ONNX kept for inspection: {build_onnx_path}")
        else:
            cleanup_tensorrt_onnx_after_engine(onnx_path, self.engine_path)
            if build_onnx_path != onnx_path:
                cleanup_tensorrt_onnx_after_engine(build_onnx_path, self.engine_path)

    def _build_engine_from_onnx(self, onnx_path: str, trt) -> None:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        builder_precision = str(
            getattr(self, "_trt_builder_precision", self.precision)
        ).strip().lower()
        strongly_typed = bool(
            builder_precision.startswith("int8")
            and getattr(self, "_trt_modelopt_int8", False)
            and _tensorrt_int8_modelopt_strongly_typed()
        )
        explicit_batch = getattr(
            getattr(trt, "NetworkDefinitionCreationFlag", object),
            "EXPLICIT_BATCH",
            None,
        )
        flags = (1 << int(explicit_batch)) if explicit_batch is not None else 0
        strong_flag = getattr(
            getattr(trt, "NetworkDefinitionCreationFlag", object),
            "STRONGLY_TYPED",
            None,
        )
        if strongly_typed and strong_flag is not None:
            flags |= 1 << int(strong_flag)
            print("TensorRT network mode: strongly typed")
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        workspace_gb = _tensorrt_workspace_gb()
        if workspace_gb is not None:
            workspace_bytes = int(max(1.0, workspace_gb) * (1024 ** 3))
            if hasattr(config, "set_memory_pool_limit"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            else:
                config.max_workspace_size = workspace_bytes
            print(f"TensorRT workspace limit: {workspace_gb:g} GiB")
        else:
            print("TensorRT workspace limit: TensorRT default")

        opt_level = _tensorrt_builder_optimization_level()
        if hasattr(config, "builder_optimization_level"):
            try:
                config.builder_optimization_level = opt_level
                print(f"TensorRT builder optimization level: {opt_level}")
            except Exception as exc:
                print(f"TensorRT builder optimization level skipped: {exc}")

        aux_streams = _tensorrt_aux_stream_count()
        if aux_streams is not None and hasattr(config, "max_aux_streams"):
            try:
                config.max_aux_streams = aux_streams
                print(f"TensorRT max auxiliary streams: {aux_streams}")
            except Exception as exc:
                print(f"TensorRT auxiliary streams skipped: {exc}")

        if hasattr(config, "profiling_verbosity") and hasattr(trt, "ProfilingVerbosity"):
            try:
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                print("TensorRT profiling verbosity: detailed")
            except Exception as exc:
                print(f"TensorRT profiling verbosity skipped: {exc}")

        native_int8 = (
            builder_precision.startswith("int8")
            and self._trt_qdq_fusion == "native"
            and not getattr(self, "_trt_modelopt_int8", False)
        )
        platform_has_fast_fp16 = bool(
            getattr(builder, "platform_has_fast_fp16", True)
        )
        platform_has_fast_int8 = bool(
            getattr(builder, "platform_has_fast_int8", True)
        )
        if (
            builder_precision != "fp32"
            and not strongly_typed
            and platform_has_fast_fp16
            and (
                _tensorrt_fp16_enabled()
                and (
                    not builder_precision.startswith("int8-full")
                    or _tensorrt_full_int8_fp16_islands_enabled()
                    or getattr(self, "_trt_modelopt_torch_int8", False)
                )
            )
        ):
            config.set_flag(trt.BuilderFlag.FP16)
        if builder_precision.startswith("int8-full"):
            if _tensorrt_full_int8_fp16_islands_enabled():
                print("TensorRT full INT8 safety: FP16 islands enabled")
            else:
                print("TensorRT full INT8 safety: all-out INT8, FP16 tactics disabled")
        if native_int8 and platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if native_int8:
                cache_path = _tensorrt_calibration_cache_path(
                    self.engine_path,
                    self._trt_calibration_cache,
                )
                calibrated_tensor_names = _read_tensorrt_calibration_cache_tensor_names(
                    cache_path
                )
                quant_flag = getattr(
                    getattr(trt, "QuantizationFlag", object),
                    "CALIBRATE_BEFORE_FUSION",
                    None,
                )
                if (
                    quant_flag is not None
                    and hasattr(config, "set_quantization_flag")
                    and _tensorrt_native_int8_calibrate_before_fusion()
                ):
                    try:
                        config.set_quantization_flag(quant_flag)
                        print("TensorRT native INT8 calibration: before fusion")
                    except Exception as exc:
                        print(f"TensorRT native INT8 calibration flag skipped: {exc}")
                elif quant_flag is not None:
                    print("TensorRT native INT8 calibration: after fusion")
                prefer_constraints = _tensorrt_native_int8_prefer_constraints()
                obey_constraints = _tensorrt_native_int8_obey_constraints()
                checkpoint_policy = {}
                policy_applied = False
                if _tensorrt_native_int8_checkpoint_policy_enabled():
                    checkpoint_policy = _tensorrt_native_int8_onnx_policy(
                        onnx_path,
                        quant_type=getattr(self, "_int8_quant_type", None),
                        w8a8_layers=getattr(self, "_int8_w8a8_layers", None),
                    )
                    int8_count, fp16_count, _ = (
                        _apply_tensorrt_native_int8_checkpoint_policy(
                            network,
                            trt,
                            checkpoint_policy,
                            int8_outputs=_tensorrt_native_int8_policy_int8_outputs(),
                            constrain_outputs=(
                                _tensorrt_native_int8_policy_output_constraints()
                            ),
                        )
                    )
                    policy_applied = bool(int8_count or fp16_count)
                    if policy_applied:
                        prefer_flag = getattr(
                            trt.BuilderFlag,
                            "PREFER_PRECISION_CONSTRAINTS",
                            None,
                        )
                        obey_flag = getattr(
                            trt.BuilderFlag,
                            "OBEY_PRECISION_CONSTRAINTS",
                            None,
                        )
                        if _tensorrt_native_int8_policy_obey() and obey_flag is not None:
                            config.set_flag(obey_flag)
                            print("TensorRT native INT8 checkpoint policy: obey")
                        elif prefer_flag is not None:
                            config.set_flag(prefer_flag)
                            print("TensorRT native INT8 checkpoint policy: prefer")
                if (prefer_constraints or obey_constraints) and not policy_applied:
                    _prefer_tensorrt_native_int8_layers(
                        network,
                        trt,
                        calibrated_tensor_names,
                    )
                    prefer_flag = getattr(
                        trt.BuilderFlag,
                        "PREFER_PRECISION_CONSTRAINTS",
                        None,
                    )
                    obey_flag = getattr(
                        trt.BuilderFlag,
                        "OBEY_PRECISION_CONSTRAINTS",
                        None,
                    )
                    if obey_constraints and obey_flag is not None:
                        config.set_flag(obey_flag)
                        print("TensorRT native INT8 constraints: obey")
                    elif prefer_flag is not None:
                        config.set_flag(prefer_flag)
                        print("TensorRT native INT8 constraints: prefer")
                if self._trt_calibration_dataset:
                    calibration_source = _make_tensorrt_dataset_calibration_source(
                        dataset_path=self._trt_calibration_dataset,
                        width=self._engine_width,
                        height=self._engine_height,
                        dtype=self._dtype,
                        device=self.device,
                        frame_count=self._trt_calibration_frames,
                    )
                elif self._trt_calibration_video and os.path.isfile(
                    self._trt_calibration_video
                ):
                    calibration_source = _make_tensorrt_video_calibration_source(
                        video_path=self._trt_calibration_video,
                        width=self._engine_width,
                        height=self._engine_height,
                        dtype=self._dtype,
                        device=self.device,
                        frame_count=self._trt_calibration_frames,
                    )
                elif os.path.isfile(cache_path):
                    calibration_source = None
                else:
                    if self._trt_calibration_video:
                        print(
                            "TensorRT calibration video missing; "
                            f"falling back to synthetic data: {self._trt_calibration_video}"
                        )
                    calibration_source = _make_synthetic_tensorrt_calibration_source(
                        width=self._engine_width,
                        height=self._engine_height,
                        dtype=self._dtype,
                        device=self.device,
                        frame_count=self._trt_calibration_frames,
                    )
                self._trt_int8_calibrator = _make_tensorrt_int8_calibrator(
                    trt,
                    cache_path=cache_path,
                    calibration_source=calibration_source,
                )
                config.int8_calibrator = self._trt_int8_calibrator
                print(
                    "TensorRT native INT8 calibration cache: "
                    f"{cache_path}"
                )

        timing_cache_path, timing_cache = self._attach_tensorrt_timing_cache(config)

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
        self._save_tensorrt_timing_cache(config, timing_cache_path)
        self._trt_engine_metadata = self._expected_tensorrt_engine_metadata(trt)
        _write_tensorrt_engine_metadata(self.engine_path, self._trt_engine_metadata)

        # Keep the cache object alive until after build_serialized_network has
        # returned; TensorRT requires it to outlive the build.
        del timing_cache

    def _attach_tensorrt_timing_cache(self, config):
        timing_cache_path = _tensorrt_timing_cache_path()
        if not timing_cache_path:
            return None, None
        if not (hasattr(config, "create_timing_cache") and hasattr(config, "set_timing_cache")):
            return timing_cache_path, None

        cache_data = b""
        try:
            if os.path.isfile(timing_cache_path):
                cache_data = pathlib.Path(timing_cache_path).read_bytes()
        except Exception as exc:
            print(f"TensorRT timing cache read skipped: {exc}")
            cache_data = b""

        try:
            timing_cache = config.create_timing_cache(cache_data)
            if timing_cache is None:
                return timing_cache_path, None
            if config.set_timing_cache(timing_cache, False):
                if cache_data:
                    print(f"TensorRT timing cache loaded: {timing_cache_path}")
                return timing_cache_path, timing_cache

            if cache_data:
                print("TensorRT timing cache rejected; starting a fresh cache.")
                timing_cache = config.create_timing_cache(b"")
                if timing_cache is not None and config.set_timing_cache(timing_cache, False):
                    return timing_cache_path, timing_cache
        except Exception as exc:
            print(f"TensorRT timing cache skipped: {exc}")
        return timing_cache_path, None

    def _save_tensorrt_timing_cache(self, config, timing_cache_path) -> None:
        if not timing_cache_path or not hasattr(config, "get_timing_cache"):
            return
        try:
            timing_cache = config.get_timing_cache()
            if timing_cache is None:
                return
            serialized = timing_cache.serialize()
            os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
            pathlib.Path(timing_cache_path).write_bytes(bytes(serialized))
            print(f"TensorRT timing cache saved: {timing_cache_path}")
        except Exception as exc:
            print(f"TensorRT timing cache save skipped: {exc}")

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
        if not self._trt_input_names or not self._trt_output_names:
            raise RuntimeError("Unexpected TensorRT bindings; expected input/output tensors.")
        if "input" in self._trt_input_names and "cond" in self._trt_input_names:
            other_inputs = [
                name for name in self._trt_input_names
                if name not in {"input", "cond"}
            ]
            self._trt_input_names = ["input", "cond"] + other_inputs
        elif len(self._trt_input_names) == 1:
            self._trt_int8_single_input = True
            print(
                "TensorRT binding discovery: single-input engine "
                f"({self._trt_input_names[0]}); condition input was pruned"
            )
        elif len(self._trt_input_names) < 2 and not getattr(
            self,
            "_trt_int8_single_input",
            False,
        ):
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
        cond = cond.contiguous() if cond is not None else None
        ctx = self._trt_context
        engine = self._trt_engine
        current_stream = torch.cuda.current_stream()
        trt_stream = self._trt_stream
        has_cond_binding = len(self._trt_input_names) > 1
        if trt_stream is not None:
            trt_stream.wait_stream(current_stream)
            tensor.record_stream(trt_stream)
            if has_cond_binding and cond is not None:
                cond.record_stream(trt_stream)
            stream = trt_stream.cuda_stream
        else:
            stream = current_stream.cuda_stream

        input_name = self._trt_input_names[0]
        cond_name = self._trt_input_names[1] if has_cond_binding else None
        output_name = self._trt_output_names[0]

        if hasattr(ctx, "set_input_shape"):
            tensor_shape = tuple(tensor.shape)
            cond_shape = tuple(cond.shape) if (has_cond_binding and cond is not None) else None
            shape_key = (tensor_shape, cond_shape) if has_cond_binding else (tensor_shape,)
            if self._trt_shape_cache_key != shape_key:
                if ctx.set_input_shape(input_name, tensor_shape) is False:
                    raise RuntimeError(f"TensorRT rejected input shape: {tensor_shape}")
                if has_cond_binding:
                    if cond is None:
                        raise RuntimeError("TensorRT engine expects cond but none was provided.")
                    if ctx.set_input_shape(cond_name, cond_shape) is False:
                        raise RuntimeError(f"TensorRT rejected cond shape: {cond_shape}")
                self._trt_shape_cache_key = shape_key
                self._trt_address_cache_key = None

            out_shape = tuple(int(v) for v in ctx.get_tensor_shape(output_name))
            if self._trt_output is None or self._trt_output_shape != out_shape:
                self._trt_output = torch.empty(
                    out_shape,
                    dtype=self._tensor_dtype_for_output(output_name),
                    device=self.device,
                )
                self._trt_output_shape = out_shape
                self._trt_address_cache_key = None
            if trt_stream is not None:
                self._trt_output.record_stream(trt_stream)

            if has_cond_binding:
                if cond is None:
                    raise RuntimeError("TensorRT engine expects cond but none was provided.")
                address_key = (
                    int(tensor.data_ptr()),
                    int(cond.data_ptr()),
                    int(self._trt_output.data_ptr()),
                )
                address_items = (
                    (input_name, address_key[0]),
                    (cond_name, address_key[1]),
                    (output_name, address_key[2]),
                )
            else:
                address_key = (
                    int(tensor.data_ptr()),
                    int(self._trt_output.data_ptr()),
                )
                address_items = (
                    (input_name, address_key[0]),
                    (output_name, address_key[1]),
                )
            if self._trt_address_cache_key != address_key:
                for name, ptr in address_items:
                    if ctx.set_tensor_address(name, int(ptr)) is False:
                        raise RuntimeError(f"TensorRT rejected tensor address: {name}")
                self._trt_address_cache_key = address_key
            ok = ctx.execute_async_v3(stream_handle=stream)
        else:
            bindings = [0] * engine.num_bindings
            input_values = [(input_name, tensor)]
            if has_cond_binding:
                if cond is None:
                    raise RuntimeError("TensorRT engine expects cond but none was provided.")
                input_values.append((cond_name, cond))
            shape_key = tuple(tuple(value.shape) for _name, value in input_values)
            shape_changed = self._trt_legacy_shape_cache_key != shape_key
            for name, value in input_values:
                idx = engine.get_binding_index(name)
                if shape_changed:
                    try:
                        ctx.set_binding_shape(idx, tuple(value.shape))
                    except Exception:
                        pass
                bindings[idx] = int(value.data_ptr())
            if shape_changed:
                self._trt_legacy_shape_cache_key = shape_key
            out_idx = engine.get_binding_index(output_name)
            out_shape = tuple(int(v) for v in ctx.get_binding_shape(out_idx))
            if self._trt_output is None or self._trt_output_shape != out_shape:
                self._trt_output = torch.empty(
                    out_shape,
                    dtype=self._tensor_dtype_for_output(output_name),
                    device=self.device,
                )
                self._trt_output_shape = out_shape
            if trt_stream is not None:
                self._trt_output.record_stream(trt_stream)
            bindings[out_idx] = int(self._trt_output.data_ptr())
            ok = ctx.execute_async_v2(bindings=bindings, stream_handle=stream)

        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        if trt_stream is not None:
            current_stream.wait_stream(trt_stream)
        return self._trt_output
