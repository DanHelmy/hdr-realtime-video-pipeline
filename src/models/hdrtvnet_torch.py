import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_HAS_COMPILE = hasattr(torch, "compile")          # PyTorch >= 2.0
_HAS_CUDA_GRAPHS = hasattr(torch.cuda, "CUDAGraph")  # PyTorch >= 1.10
_IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
_IS_NVIDIA = torch.cuda.is_available() and not _IS_ROCM

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Auto-detect HIP SDK on ROCm-Windows (needed for Triton codegen).
# Checks HIP_PATH env or the standard AMD ROCm install directory.
def _detect_hip_sdk():
    if os.name != "nt" or not _IS_ROCM:
        return False  # only relevant on ROCm-Windows
    hip_path = os.environ.get("HIP_PATH", "")
    if hip_path and os.path.isdir(os.path.join(hip_path, "include", "hip")):
        return True
    # Check standard install locations
    rocm_root = r"C:\Program Files\AMD\ROCm"
    if os.path.isdir(rocm_root):
        for entry in os.listdir(rocm_root):
            candidate = os.path.join(rocm_root, entry, "include", "hip")
            if os.path.isdir(candidate):
                return True
    return False

_HAS_HIP_SDK = _detect_hip_sdk()

# Enable TF32 on Ampere+ GPUs — harmless no-op on AMD
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
# Full INT8 (W8A8) replacement layers — weights AND activations INT8
# ===================================================================

class W8A8Conv2d(nn.Module):
    """Conv2d with INT8 weights and INT8 activations (static quantization).

    Weights use per-output-channel scale.  Activations use per-tensor scale
    determined during a calibration pass.  At inference both are dequantized
    to ``compute_dtype`` and passed to F.conv2d.  torch.compile fuses the
    dequant + conv into a single efficient kernel.
    """

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
        # Activation scale — set during calibration via calibrate_w8a8()
        self.register_buffer("x_scale", torch.tensor(1.0, dtype=compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activation to INT8 then dequantize
        x_int8 = (x / self.x_scale).round().clamp(-128, 127).to(torch.int8)
        x_deq = x_int8.to(self.compute_dtype) * self.x_scale
        # Dequantize weights
        w = self.weight_int8.to(self.compute_dtype) * self.w_scale.view(-1, 1, 1, 1)
        return F.conv2d(x_deq, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class W8A8Linear(nn.Module):
    """Linear with INT8 weights and INT8 activations (static quantization)."""

    def __init__(self, linear: nn.Linear, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int8 = (x / self.x_scale).round().clamp(-128, 127).to(torch.int8)
        x_deq = x_int8.to(self.compute_dtype) * self.x_scale
        w = self.weight_int8.to(self.compute_dtype) * self.w_scale.view(-1, 1)
        return F.linear(x_deq, w, self.bias)


def _quantize_model_w8a8(model: nn.Module,
                         compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    """Replace all Conv2d and Linear with W8A8 equivalents (in-place)."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            if isinstance(module, nn.Conv2d):
                setattr(parent, parts[-1], W8A8Conv2d(module, compute_dtype))
            else:
                setattr(parent, parts[-1], W8A8Linear(module, compute_dtype))
            replaced += 1
    print(f"  Quantized {replaced} layers to W8A8 (INT8 weights + activations, "
          f"{compute_dtype} compute)")
    return model


def _is_memory_bound_1x1(module: nn.Conv2d, channel_threshold: int = 32) -> bool:
    """Decide if a Conv2d is a memory-bound 1×1 that benefits from INT8 IO.

    Our Triton benchmark showed 1.74-2.28× speedup for 1×1 convs with
    max(C_in, C_out) ≤ 32 (SFT layers) but 0.54-0.91× for larger channels
    (AGCM / condition network).  Following the FSR4 approach of using INT8
    as *memory compression* rather than compute format, we apply W8A8 only
    to these bandwidth-sensitive layers.
    """
    if module.kernel_size != (1, 1):
        return False
    return max(module.in_channels, module.out_channels) <= channel_threshold


def _quantize_model_mixed(model: nn.Module,
                          compute_dtype: torch.dtype = torch.float16,
                          channel_threshold: int = 32) -> nn.Module:
    """Selective mixed INT8: W8A8 for memory-bound 1×1 convs, W8A16 otherwise.

    Strategy inspired by FSR4's INT8 path (DP4A-based memory compression):
      * SFT 1×1 convs (≤32 ch) → W8A8  (INT8 weights + activations)
      * 3×3 convs / large 1×1s  → W8A16 (INT8 weights only)
      * Linear layers            → W8A16 (INT8 weights only)
    """
    w8a8_count = 0
    w8_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            if isinstance(module, nn.Conv2d) and _is_memory_bound_1x1(
                    module, channel_threshold):
                setattr(parent, parts[-1], W8A8Conv2d(module, compute_dtype))
                w8a8_count += 1
            elif isinstance(module, nn.Conv2d):
                setattr(parent, parts[-1], W8Conv2d(module, compute_dtype))
                w8_count += 1
            else:
                setattr(parent, parts[-1], W8Linear(module, compute_dtype))
                w8_count += 1
    total = w8a8_count + w8_count
    print(f"  Mixed INT8: {w8a8_count} layers → W8A8, {w8_count} layers → W8A16 "
          f"({total} total, {compute_dtype} compute)")
    return model


def calibrate_w8a8(model: nn.Module, calibration_inputs: list) -> None:
    """Run calibration data through the model to determine activation scales.

    Each W8A8Conv2d / W8A8Linear records the maximum absolute activation value
    observed across all calibration samples, then sets x_scale = max_abs / 127.
    """
    # Attach hooks to record activation ranges
    hooks = []
    max_vals = {}

    def _make_hook(name):
        def _hook(module, inp, out):
            x = inp[0]
            abs_max = x.detach().abs().amax().item()
            if name not in max_vals or abs_max > max_vals[name]:
                max_vals[name] = abs_max
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, (W8A8Conv2d, W8A8Linear)):
            hooks.append(module.register_forward_hook(_make_hook(name)))

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
    for name, module in model.named_modules():
        if isinstance(module, (W8A8Conv2d, W8A8Linear)) and name in max_vals:
            x_scale = max(max_vals[name], 1e-8) / 127.0
            module.x_scale.fill_(x_scale)
            calibrated += 1

    print(f"  Calibrated {calibrated} layers (activation scales set)")


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
      * NVIDIA: cudnn.benchmark + channels_last (auto).
      * torch.compile with max-autotune (auto when Triton + HIP SDK present).
      * AMD ROCm-Windows: auto-detects HIP SDK; use --force-compile to
        override if detection fails.
      * Optional CUDA-graph replay for static-shape inputs.
    """

    def __init__(self, model_path, device="auto", precision="auto",
                 compile_model=True, force_compile=False, compile_mode="auto",
                 use_cuda_graphs=False, force_channels_last=False):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.precision = self._resolve_precision(precision, self.device)
        self._use_cuda = self.device.type == "cuda"
        self._dtype = {"fp16": torch.float16}.get(
            self.precision, torch.float32
        )
        self._is_flat_model = False  # True for old-style FX quantized models
        self._is_w8_model = False    # True for GPU W8 weight-only INT8

        # ---- Print platform info ------------------------------------------
        if self._use_cuda:
            gpu_name = _get_gpu_info()
            backend = "ROCm" if _IS_ROCM else "CUDA"
            print(f"GPU: {gpu_name} ({backend})")

        self.model = self._load_model(model_path)

        # ---- Platform-specific: channels_last + cudnn.benchmark -----------
        # Auto-enabled on NVIDIA (cuDNN benefits); skipped on ROCm where
        # MIOpen NHWC can hurt.  --channels-last forces it on for A/B testing.
        if self._use_cuda and (_IS_NVIDIA or force_channels_last):
            if _IS_NVIDIA:
                torch.backends.cudnn.benchmark = True
            self.model = self.model.to(memory_format=torch.channels_last)
            self._use_channels_last = True
            if _IS_ROCM:
                print("ROCm + --channels-last: channels_last forced ON")
            else:
                print("NVIDIA: cudnn.benchmark + channels_last enabled")
        else:
            self._use_channels_last = False
            if self._use_cuda and _IS_ROCM:
                print("ROCm: skipping channels_last (MIOpen works better "
                      "without it for this model). Use --channels-last to override.")

        # ---- torch.compile (PyTorch 2.x) ----------------------------------
        # NVIDIA / ROCm-Linux: auto-enabled when Triton is available.
        # ROCm-Windows: auto-enabled when HIP SDK is detected; use
        #   --force-compile to override if auto-detection misses it.
        self._compiled = False
        _rocm_win_no_sdk = (
            _IS_ROCM and os.name == "nt"
            and not force_compile and not _HAS_HIP_SDK
        )
        if compile_model and _HAS_COMPILE and self._use_cuda:
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
                "  Re-run: python quantize_int8_full.py or quantize_int8_mixed.py"
            )
        arch = checkpoint.get("architecture", {})
        compute_dtype_str = checkpoint.get("compute_dtype", "torch.float16")
        compute_dtype = torch.float16 if "16" in compute_dtype_str else torch.float32
        self._dtype = compute_dtype
        quant_type = checkpoint.get("quantization", "w8_weight_only")

        # Build base architecture
        model = Ensemble_AGCM_LE(
            classifier=arch.get("classifier", "color_condition"),
            cond_c=arch.get("cond_c", 6),
            in_nc=arch.get("in_nc", 3),
            out_nc=arch.get("out_nc", 3),
            nf=arch.get("nf", 32),
            act_type=arch.get("act_type", "relu"),
            weighting_network=arch.get("weighting_network", False),
        )

        # Replace Conv2d / Linear with correct quantized equivalents
        if quant_type == "w8a8_mixed":
            _quantize_model_mixed(model, compute_dtype,
                                  channel_threshold=checkpoint.get(
                                      "channel_threshold", 32))
        elif quant_type == "w8a8_full":
            _quantize_model_w8a8(model, compute_dtype)
        else:
            raise ValueError(
                f"Unknown quantization type '{quant_type}' in checkpoint.\n"
                "  Supported: w8a8_full, w8a8_mixed")

        # Load the quantized state_dict
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        # Cast ALL remaining layers (InstanceNorm, etc.) to compute_dtype.
        # On ROCm, InstanceNorm in FP32 triggers broken MIOpen JIT compilation.
        # On CUDA, this is a harmless optimization (avoids unnecessary casts).
        model = model.to(dtype=compute_dtype, device=self.device)
        model.eval()
        self._is_w8_model = True
        label = {"w8a8_full": "W8A8", "w8a8_mixed": "Mixed W8A8/W8A16"}.get(
            quant_type, quant_type)
        print(f"Loaded {label} INT8 model (compute_dtype={compute_dtype})")
        return model

    def _load_model(self, model_path):
        # INT8 quantized models use a different loading path
        if self.precision in ("int8-full", "int8-mixed"):
            return self._load_int8_model(model_path)

        ext = os.path.splitext(model_path)[1].lower()
        if ext in {".pt", ".ts"}:
            model = torch.jit.load(model_path, map_location=self.device)
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
    # Buffer management — allocate once, reuse every frame
    # -----------------------------------------------------------------------
    def _ensure_buffers(self, h, w):
        """Allocate / reallocate persistent GPU tensors when resolution changes.
        For the common case (fixed resolution) this is a no-op after the first
        frame."""
        if self._buf_hw == (h, w):
            return
        self._buf_hw = (h, w)
        cond_h, cond_w = max(1, h // 4), max(1, w // 4)

        # Persistent GPU tensors — avoids torch.empty() + .to(device) per frame
        mem_fmt = (torch.channels_last if self._use_channels_last
                   else torch.contiguous_format)
        self._gpu_input = torch.empty(
            (1, 3, h, w), dtype=self._dtype, device=self.device,
        ).to(memory_format=mem_fmt)
        self._gpu_cond = torch.empty(
            (1, 3, cond_h, cond_w), dtype=self._dtype, device=self.device,
        ).to(memory_format=mem_fmt)

        # Invalidate any cached CUDA graph on resolution change
        self._graph = None
        self._graph_hw = None

    # -----------------------------------------------------------------------
    # Preprocess — keep as much work on GPU as possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._ensure_buffers(h, w)

        # Upload raw uint8 BGR frame to GPU, then do all math there
        # from_numpy is zero-copy; the .to(device) does the actual H2D
        raw = torch.from_numpy(frame_bgr).to(
            device=self.device, non_blocking=self._use_cuda
        )
        # (H,W,3) uint8 → BGR→RGB via channel flip → CHW → add batch → fp → /255
        raw = raw.flip(2)                                  # BGR → RGB
        raw = raw.permute(2, 0, 1).unsqueeze(0)           # (1,3,H,W)
        self._gpu_input.copy_(
            raw.to(dtype=self._dtype).mul_(1.0 / 255.0),
            non_blocking=self._use_cuda,
        )

        # Condition tensor (0.25× spatial)
        self._gpu_cond.copy_(
            F.interpolate(
                self._gpu_input,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ),
            non_blocking=self._use_cuda,
        )

        return self._gpu_input, self._gpu_cond

    # -----------------------------------------------------------------------
    # Inference — CUDA graph replay when possible
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
    # Postprocess — keep as much work on GPU as possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def postprocess(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]

        # All math on GPU: clamp, scale, quantize, channel-flip (RGB→BGR)
        t = output.squeeze(0)                      # (3,H,W)  fp16/fp32
        t = t.clamp_(0.0, 1.0).mul_(255.0)        # [0,255] still on GPU
        t = t.to(dtype=torch.uint8)                # quantize on GPU
        t = t.flip(0)                              # RGB→BGR via channel flip
        t = t.permute(1, 2, 0).contiguous()        # CHW → HWC, contiguous for cv2
        return t.cpu().numpy()                     # single D2H copy of uint8

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

    def end_profiling(self):
        return None
